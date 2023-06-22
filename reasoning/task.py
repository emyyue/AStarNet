from ogb import linkproppred

import torch
from torch.utils import data as torch_data

from torchdrug import core, tasks
from torchdrug.core import Registry as R

from torch.nn import functional as F
import torch_scatter
from torchdrug.layers import functional
from torchdrug import core, tasks, metrics

@R.register("task.InductiveKnowledgeGraphCompletion")
class InductiveKnowledgeGraphCompletion(tasks.KnowledgeGraphCompletion, core.Configurable):

    def __init__(self, model, criterion="bce", metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True, sample_weight=True,
                 full_batch_eval=False):
        super(InductiveKnowledgeGraphCompletion, self).__init__(
            model, criterion, metric, num_negative, margin, adversarial_temperature, strict_negative,
            sample_weight=sample_weight, filtered_ranking=True, full_batch_eval=full_batch_eval)

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.register_buffer("graph", dataset.graph)
        self.register_buffer("fact_graph", dataset.fact_graph)
        self.register_buffer("inductive_graph", dataset.inductive_graph)
        self.register_buffer("inductive_fact_graph", dataset.inductive_fact_graph)

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)

    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)
        graph = self.inductive_fact_graph if self.split == "test" else self.fact_graph

        if all_loss is None:
            # test
            all_index = torch.arange(graph.num_node, device=self.device)
            num_negative = graph.num_node if self.full_batch_eval else self.num_negative
            t_preds = []
            h_preds = []
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

        return pred

    def target(self, batch):
        # test target
        batch_size = len(batch)
        graph = self.inductive_graph if self.split == "test" else self.graph
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = graph.match(pattern)
        t_truth_index = graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = graph.match(pattern)
        h_truth_index = graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        # in case of GPU OOM
        return mask.cpu(), target.cpu()


@R.register("task.KnowledgeGraphCompletionOGB")
class KnowledgeGraphCompletionOGB(tasks.KnowledgeGraphCompletion, core.Configurable):

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.evaluator = linkproppred.Evaluator(dataset.name)
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        fact_mask = torch.zeros(len(dataset), dtype=torch.bool)
        fact_mask[train_set.indices] = 1
        if self.fact_ratio:
            length = int(len(train_set) * self.fact_ratio)
            index = torch.randperm(len(train_set))[length:]
            train_indices = torch.tensor(train_set.indices)
            fact_mask[train_indices[index]] = 0
            train_set = torch_data.Subset(train_set, index)
        self.register_buffer("fact_graph", dataset.graph.edge_mask(fact_mask))

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)

        return train_set, valid_set, test_set

    def predict(self, batch, all_loss=None, metric=None):
        batch_size = len(batch)

        if all_loss is None:
            # test
            h_index, t_index, r_index = batch.unbind(-1)
            pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

        return pred

    def target(self, batch):
        # test target
        batch_size = len(batch)
        target = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # in case of GPU OOM
        return target.cpu()

    def evaluate(self, pred, target):
        is_positive = torch.zeros(pred.shape, dtype=torch.bool)
        is_positive.scatter_(-1, target.unsqueeze(-1), 1)
        pos_pred = pred[is_positive]
        neg_pred = pred[~is_positive].view(len(pos_pred), -1)
        metric = self.evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})

        new_metric = {}
        for key in metric:
            new_key = key.split("_")[0]
            new_metric[new_key] = metric[key].mean()

        return new_metric
    
    
@R.register("tasks.KnowledgeGraphCompletionBiomed")
class KnowledgeGraphCompletionBiomed(tasks.KnowledgeGraphCompletion, core.Configurable):

    def __init__(self, model, criterion="bce",
                 metric=("mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@100", "auroc", "ap"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True,
                 heterogeneous_negative=False, heterogeneous_evaluation=False, filtered_ranking=True,
                 fact_ratio=None, sample_weight=True, conditional_probability=False,
                 full_batch_eval=False):
        super(KnowledgeGraphCompletionBiomed, self).__init__(model=model, criterion=criterion, metric=metric, 
                                                             num_negative=num_negative, margin=margin,
                                                             adversarial_temperature=adversarial_temperature, 
                                                             strict_negative=strict_negative,
                                                             filtered_ranking=filtered_ranking,fact_ratio=fact_ratio,
                                                             sample_weight=sample_weight, full_batch_eval=full_batch_eval)
        self.heterogeneous_negative = heterogeneous_negative
        self.heterogeneous_evaluation = heterogeneous_evaluation
        self.conditional_probability = conditional_probability
        
    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.register_buffer("graph", dataset.graph)
        fact_mask = torch.ones(len(dataset), dtype=torch.bool)
        fact_mask[valid_set.indices] = 0
        fact_mask[test_set.indices] = 0
        if self.fact_ratio:
            length = int(len(train_set) * self.fact_ratio)
            index = torch.randperm(len(train_set))[length:]
            train_indices = torch.tensor(train_set.indices)
            fact_mask[train_indices[index]] = 0
            train_set = torch_data.Subset(train_set, index)
        self.register_buffer("fact_graph", dataset.graph.edge_mask(fact_mask))

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)
            
        self.register_buffer("undirected_fact_graph", self.fact_graph.undirected(add_inverse=True))
        with self.undirected_fact_graph.graph():
            self.undirected_fact_graph.degree_in_type = self.get_degree_in_type(self.undirected_fact_graph)
            self.undirected_fact_graph.num_nodes_per_type = torch.bincount(self.undirected_fact_graph.node_type)
                    
        return train_set, valid_set, test_set
        

    def get_degree_in_type(self, graph):        
        ########################
        # making degree_in_type based on relations, as same nodes might have different relation types
        ########################

        # count the number of occurrence for each relation type for each node
        myindex = graph.edge_list[:, 0]
        relation_type = graph.edge_list[:, 2]
        # one hot encoding of relation types
        # Zhaocheng: myinput is (|E|, |R|), potentially OOM for large graphs
        # You may augment myindex to be relation_type * graph.num_node + my_index
        # and scatter_add ones with the augmented index
        # finally reshape the tensor from (num_relation * num_node,) to (num_relation, num_node)
        myinput = torch.t(F.one_hot(relation_type))
        # calculate
        degree_in_type = myinput.new_zeros(graph.num_relation,  graph.num_node) # which output dim
        degree_in_type = torch_scatter.scatter_add(myinput, myindex, out=degree_in_type)
        
        return degree_in_type
        
        
    def target(self, batch):
        # test target
        batch_size = len(batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        any = -torch.ones_like(pos_h_index)
        node_type = self.fact_graph.node_type

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = self.graph.match(pattern)
        t_truth_index = self.graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(batch_size, self.num_entity, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0
        if self.heterogeneous_evaluation:
            t_mask[node_type.unsqueeze(0) != node_type[pos_t_index].unsqueeze(-1)] = 0

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = self.graph.match(pattern)
        h_truth_index = self.graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(batch_size, self.num_entity, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0
        if self.heterogeneous_evaluation:
            h_mask[node_type.unsqueeze(0) != node_type[pos_h_index].unsqueeze(-1)] = 0

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        # in case of GPU OOM
        return mask.cpu(), target.cpu()

    def evaluate(self, pred, target):
        
        if self.conditional_probability:
            mask, target = target

            pos_pred = pred.gather(-1, target.unsqueeze(-1))
            if self.filtered_ranking:
                ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
            else:
                ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
            
            # get neg predictions
            # remove the true tail and head
            m = torch.ones_like(pred).scatter(2, target.unsqueeze(-1), 0).bool()
            # use mask to get rid of other true tails and heads
            prob =  m.logical_and(~mask).long().float()
            # sample from neg exampels
            neg_t = functional.multinomial(prob[:,0,:], 1, replacement=True)
            neg_h = functional.multinomial(prob[:,1,:], 1, replacement=True)
            # concat and get mask
            mask_neg =  torch.cat((neg_t, neg_h), -1)
            # get neg predictions
            neg_pred = pred.gather(-1, mask_neg.unsqueeze(-1))
            # take random sample of neg predictions
            pred = torch.stack((pos_pred.flatten(), neg_pred.flatten()),1)
            # construct the target out of positive (1) and negative (0)
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            pred = pred.flatten()
            target = target.flatten()
            
            metric = {}
            
            for _metric in self.metric:
                if _metric == "mr":
                    score = ranking.float().mean()
                elif _metric == "mrr":
                    score = (1 / ranking.float()).mean()
                elif _metric.startswith("hits@"):
                    threshold = int(_metric[5:])
                    score = (ranking <= threshold).float().mean()
                elif _metric == "auroc":
                    score = metrics.area_under_roc(pred, target)
                elif _metric == "ap":
                    score = metrics.area_under_prc(pred, target)
                else:
                    continue
                    #raise ValueError("Unknown metric `%s`" % _metric)
                name = tasks._get_metric_name(_metric)
                metric[name] = score
        else:
            # joint
            pred = pred[:,0:2]
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            pred = pred.flatten()
            target = target.flatten()
            metric = {}
            for _metric in self.metric:
                if _metric == "auroc":
                    score = metrics.area_under_roc(pred, target)
                elif _metric == "ap":
                    score = metrics.area_under_prc(pred, target)
                else:
                    continue
                    #raise ValueError("Unknown metric `%s`" % _metric)
                name = tasks._get_metric_name(_metric)
                metric[name] = score
            
        return metric
    
    def predict(self, batch, dataset, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)

        if all_loss is None:
            # test
            if self.conditional_probability:
                # conditional probability
                all_index = torch.arange(self.num_entity, device=self.device) # evaluate against all nodes
                t_preds = []
                h_preds = []
                num_negative = self.num_entity if self.full_batch_eval else self.num_negative
                # Zhaocheng: Do you want to evaluate in both directions or just a single direction?
                # Emy: actually, mostly, only one direction, but keep it general?
                for neg_index in all_index.split(num_negative):
                    r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                    h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                    t_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric, conditional_probability=self.conditional_probability)
                    t_preds.append(t_pred)
                t_pred = torch.cat(t_preds, dim=-1)
                for neg_index in all_index.split(num_negative):
                    r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                    t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                    h_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric, conditional_probability=self.conditional_probability)
                    h_preds.append(h_pred)
                    
                h_pred = torch.cat(h_preds, dim=-1)
                pred = torch.stack([t_pred, h_pred], dim=1)
                # in case of GPU OOM
                pred = pred.cpu()
                
            else:
            # joint probability
                graph = self.undirected_fact_graph
                num_nodes_per_type = graph.num_nodes_per_type
                degree_in_type = graph.degree_in_type
                
                if self.strict_negative:
                    neg_h_index, neg_t_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index, degree_in_type, num_nodes_per_type, graph)
                else:
                    neg_h_index, neg_t_index = torch.randint(self.num_node, (2, batch_size * batch_size * self.num_negative), device=self.device)
                neg_h_index = neg_h_index.view(batch_size, self.num_negative)
                neg_t_index = neg_t_index.view(batch_size, self.num_negative)
                h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                h_index[:, 1:] = neg_h_index
                t_index[:, 1:] = neg_t_index
                pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric, conditional_probability = self.conditional_probability)

        else:
            # train
            if self.conditional_probability:
                # conditional probability
                if self.strict_negative:
                    neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
                else:
                    neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
                h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
                h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
                pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric, conditional_probability = self.conditional_probability)
            else:                
                graph = self.undirected_fact_graph
                num_nodes_per_type = graph.num_nodes_per_type
                degree_in_type = graph.degree_in_type
                
                # sample negative samples
                if self.strict_negative:
                    neg_h_index, neg_t_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index, degree_in_type, num_nodes_per_type, graph)
                else:
                    neg_h_index, neg_t_index = torch.randint(self.num_node, (2, batch_size * self.num_negative), device=self.device)
                neg_h_index = neg_h_index.view(batch_size, self.num_negative)
                neg_t_index = neg_t_index.view(batch_size, self.num_negative)
                h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                h_index[:, 1:] = neg_h_index
                t_index[:, 1:] = neg_t_index
                pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric, conditional_probability = self.conditional_probability)
            
        return pred
    

    
    @torch.no_grad()
    def _strict_negative(self, pos_h_index, pos_t_index, pos_r_index, degree_in_type=None, num_nodes_per_type=None, graph=None):
        if self.conditional_probability:
            # conditional probaility - classical KG setting

            batch_size = len(pos_h_index)
            any = -torch.ones_like(pos_h_index)
            node_type = self.fact_graph.node_type

            pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
            pattern = pattern[:batch_size // 2]
            edge_index, num_t_truth = self.fact_graph.match(pattern)
            t_truth_index = self.fact_graph.edge_list[edge_index, 1]
            pos_index = torch.repeat_interleave(num_t_truth)
            if self.heterogeneous_negative:
                pos_t_type = node_type[pos_t_index[:batch_size // 2]]
                t_mask = pos_t_type.unsqueeze(-1) == node_type.unsqueeze(0)
            else:
                t_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
            t_mask[pos_index, t_truth_index] = 0
            neg_t_candidate = t_mask.nonzero()[:, 1]
            num_t_candidate = t_mask.sum(dim=-1)
            neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, self.num_negative)

            pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
            pattern = pattern[batch_size // 2:]
            edge_index, num_h_truth = self.fact_graph.match(pattern)
            h_truth_index = self.fact_graph.edge_list[edge_index, 0]
            pos_index = torch.repeat_interleave(num_h_truth)
            if self.heterogeneous_negative:
                pos_h_type = node_type[pos_h_index[batch_size // 2:]]
                h_mask = pos_h_type.unsqueeze(-1) == node_type.unsqueeze(0)
            else:
                h_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
            h_mask[pos_index, h_truth_index] = 0
            neg_h_candidate = h_mask.nonzero()[:, 1]
            num_h_candidate = h_mask.sum(dim=-1)
            neg_h_index = functional.variadic_sample(neg_h_candidate, num_h_candidate, self.num_negative)

            neg_index = torch.cat([neg_t_index, neg_h_index])
            return neg_index
        
        else:            
            # joint probaility - rank each positive against negative samples from the same entity types as the positive ones
            
            # assert not none
            assert degree_in_type is not None
            assert num_nodes_per_type is not None
            assert graph is not None 
            node_type = graph.node_type
            
            ####################### 
            # sample from p(h)
            #######################

            # find the node types of pos_t
            pos_t_type = node_type[pos_t_index]
            pos_h_type = node_type[pos_h_index]

            pos_r_index_rev = (pos_r_index + self.num_relation) % (self.num_relation * 2)
            prob = ((num_nodes_per_type[pos_t_type]*2).unsqueeze(1) - 
                    (degree_in_type[pos_r_index] + degree_in_type[pos_r_index_rev])).float()

            same_type_mask = pos_t_type == pos_h_type
            prob[same_type_mask] -= 1
            # set to 0, if not from desired node type
            h_mask = node_type.unsqueeze(0) != pos_h_type.unsqueeze(1)
            prob[h_mask] = 0     
            

            # sample from the distribution
            neg_h_index = functional.multinomial(prob, self.num_negative, replacement=True)
            neg_h_index = torch.flatten(neg_h_index)
            
            # assert if correct node type of neg_h_index
            neg_h_type = node_type[neg_h_index]
            node_type_neg_h_bool = (neg_h_type.view(len(pos_h_index), self.num_negative)) == pos_h_type.unsqueeze(-1)
            assert torch.all(node_type_neg_h_bool)

            ####################### 
            # sample from p(t|h)
            #######################
            any = -torch.ones_like(neg_h_index)
            
            # find all the edges from neg_h_index that EXISTS
            pattern = torch.stack([neg_h_index, any, pos_r_index.repeat_interleave(self.num_negative)], dim=-1)
            edge_index, num_t_truth = graph.match(pattern)
            t_truth_index = graph.edge_list[edge_index, 1]
        
            pos_index = torch.repeat_interleave(num_t_truth)
            
            # heterogeneous
            if self.heterogeneous_negative:
                pos_t_type = node_type[pos_t_index].repeat_interleave(self.num_negative)
                t_mask = pos_t_type.unsqueeze(-1) == node_type.unsqueeze(0)
            else:
                t_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
            
            # exclude those that exists
            t_mask[pos_index, t_truth_index] = 0
            t_mask.scatter_(1, neg_h_index.unsqueeze(-1), 0)
            neg_t_candidate = t_mask.nonzero()[:, 1]
            num_t_candidate = t_mask.sum(dim=-1)
            neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, 1).squeeze(-1)
            
            # assert if correct node type of neg_t_index
            neg_t_type = node_type[neg_t_index]
            pos_t_type = node_type[pos_t_index]
            node_type_neg_t_bool = (neg_t_type.view(len(pos_h_index), self.num_negative)) == pos_t_type.unsqueeze(-1)
            assert torch.all(node_type_neg_t_bool)
            
            return neg_h_index, neg_t_index
    
    
