import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.projects.point_rend.point_features import point_sample

class Edge:
    """边结构体"""
    def __init__(self, to, next, flow, dis, from_):
        self.to = to  # 目标节点
        self.next = next  # 起点
        self.flow = flow  # 流量
        self.dis = dis  # 花费
        self.from_ = from_

class MinCostMaxFlow(nn.Module):
    """最小费用最大流算法类"""
    def __init__(self, n_classes, uni_classes, n_points=12544, ignore_lb=255):
        super(MinCostMaxFlow, self).__init__()
        self.uni_classes = uni_classes
        n = n_classes + self.uni_classes + 2
        m = n_classes + self.uni_classes + 5 * n_classes 
        s = 0
        t = n-1
        self.vis = np.zeros(n, dtype=bool)  # 记录节点是否被访问过
        self.dis = np.full(n, np.inf)  # 最小花费
        self.pre = np.zeros(n, dtype=int)  # 每个点的前驱
        self.last = np.zeros(2*m, dtype=int)  # 每个点的所连的前一条边
        self.flow = np.full(n, np.inf)  # 源点到此处的流量
        self.head = np.full(n, -1, dtype=int)  # 头指针数组
        self.num_edge = 0  # 边数
        self.maxflow = 0  # 最大流量
        self.mincost = 0  # 最小花费
        self.q = []  # 队列
        self.edge = [None] * 2*m  # 边数组

        self.n = n  # 节点数
        self.m = m  # 边数
        self.s = s  # 源点
        self.t = t  # 汇点        

        self.num_points = n_points
        self.n_classes = n_classes
        
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none') #torch.nn.BCELoss(reduction='none')
        self.ignore_lb = ignore_lb
        
    
    def construct_edges(self, unify_logits, target, bipart):
        bs = unify_logits.shape[0]
        # unify_logits = unify_logits
        # target = target
        # bipart = bipart

        point_coords = torch.rand(1, self.num_points, 2, device=unify_logits.device, dtype=unify_logits.dtype)
        # get gt labels
        tgt_mask = point_sample(
            target.unsqueeze(1).to(unify_logits.dtype),
            point_coords.repeat(target.shape[0], 1, 1),
            padding_mode='reflection',
            mode='nearest'
        ).squeeze(1)  #.view(-1)
        

        out_mask = point_sample(
            unify_logits,
            point_coords.repeat(target.shape[0], 1, 1),
            align_corners=False,
            padding_mode='reflection'
        )
        out_mask = F.softmax(out_mask, dim=1)
        
        self.link_num = 0
        for i in range(self.n_classes):
            tgt = tgt_mask == i
            if not tgt.any():
                continue
            tgt = tgt.to(unify_logits.dtype)
            losses = self.bceloss(out_mask.view(-1, self.num_points)[None], tgt.repeat(1, self.uni_classes, 1))
            
            costs = []
            for j in range(self.uni_classes):
                loss = torch.mean(losses[:, j, :])
                cost = (1-bipart[i, j]) + loss
                costs.append(cost)
            values, indexs = torch.tensor(costs).topk(5, largest=False)
            self.link_num += 5
            for idx, v in zip(indexs, values.cpu()):
                self.add_edge(1+idx, 1+self.uni_classes+i, 1, v)
                self.add_edge(1+self.uni_classes+i, 1+idx,  0, -v)
                
        for j in range(self.uni_classes):
            self.add_edge(0, 1+j, 1, 0)
            self.add_edge(1+j, 0, 0, 0)
        
        for i in range(self.n_classes):
            self.add_edge(1+self.uni_classes+i, self.n-1, 2, 0)
            self.add_edge(self.n-1, 1+self.uni_classes+i, 0, 0)
                    
            
        
    def init_by_nmst(self, n, m, s, t):
        self.vis = np.zeros(n, dtype=bool)  # 记录节点是否被访问过
        self.dis = np.full(n, np.inf)  # 最小花费
        self.pre = np.zeros(n, dtype=int)  # 每个点的前驱
        self.last = np.zeros(2*m, dtype=int)  # 每个点的所连的前一条边
        self.flow = np.full(n, np.inf)  # 源点到此处的流量
        self.head = np.full(n, -1, dtype=int)  # 头指针数组
        self.num_edge = 0  # 边数
        self.maxflow = 0  # 最大流量
        self.mincost = 0  # 最小花费
        self.q = []  # 队列
        self.edge = [None] * 2*m  # 边数组

        self.n = n  # 节点数
        self.m = m  # 边数
        self.s = s  # 源点
        self.t = t  # 汇点

    def add_edge(self, from_, to, flow, dis):
        """添加边"""
        self.edge[self.num_edge] = Edge(to, self.head[from_], flow, dis, from_)
        self.head[from_] = self.num_edge
        self.num_edge += 1
        
        

    def spfa(self, s, t):
        """SPFA算法"""
        self.dis.fill(np.inf)
        self.flow.fill(np.inf)
        self.vis.fill(False)
        self.q.append(s)
        self.vis[s] = True
        self.dis[s] = 0
        self.pre[t] = -1

        while self.q:
            now = self.q.pop(0)
            self.vis[now] = False
            i = self.head[now]
            while i != -1:
                if self.edge[i].flow > 0 and self.dis[self.edge[i].to] > self.dis[now] + self.edge[i].dis:
                    self.dis[self.edge[i].to] = self.dis[now] + self.edge[i].dis
                    self.pre[self.edge[i].to] = now
                    self.last[self.edge[i].to] = i
                    self.flow[self.edge[i].to] = min(self.flow[now], self.edge[i].flow)
                    if not self.vis[self.edge[i].to]:
                        self.vis[self.edge[i].to] = True
                        self.q.append(self.edge[i].to)
                i = self.edge[i].next

        return self.pre[t] != -1

    def forward(self, unify_logits, target, bipart):
        self.construct_edges(unify_logits, target, bipart)
        """最小费用最大流算法"""
        while self.spfa(self.s, self.t):
            now = self.t
            self.maxflow += self.flow[self.t]
            self.mincost += self.flow[self.t] * self.dis[self.t]
            while now != self.s:
                self.edge[self.last[now]].flow -= self.flow[self.t]
                self.edge[self.last[now] ^ 1].flow += self.flow[self.t]
                now = self.pre[now]
        ret = 255 * torch.ones(self.uni_classes)
        for i in range(0, 2*self.link_num, 2):
            if self.edge[i].flow == 0:
                ret[self.edge[i].from_-1] = self.edge[i].to-1-self.uni_classes
        return ret
                
            

if __name__ == "__main__":
    # n, m, s, t = map(int, input().split())
    import time
    
    n_classes = 2
    unify_logits = torch.tensor([[[[1,1],
                                  [0,0]],
                                 [[1.5,0.1],
                                  [0,0]],
                                 [[1,1],
                                  [1,1]],
                                 [[0,0],
                                  [1,1]],
                                 [[0.2,1.5],
                                  [0,0]]]])
    target = torch.tensor([[[0,0],
                           [1,1]]])
    bipart = torch.tensor([[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2]])
    T1 = time.time()
    # 初始化最小费用最大流算法类
    mcmf = MinCostMaxFlow(2, unify_logits, target, bipart, 16)

    # 求解最小费用最大流
    print(mcmf.mcmf())
    T2 = time.time()
    # 输出最大流量和最小花费
    print(int(mcmf.maxflow), int(mcmf.mincost))
    for e in mcmf.edge:
        print(f"edge {e.to}, dis: {e.dis}, flow: {e.flow}")
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
