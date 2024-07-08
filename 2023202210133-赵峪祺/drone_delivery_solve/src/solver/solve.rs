use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use itertools::Itertools;
use ordered_float::OrderedFloat;

use super::{graph::{AdjMat, Edge, Point, PointRole}, order::{pri_to_num, print_orders, Order, Priority}, tree::Tree};

#[derive(Clone, Copy)]
pub struct Config {
    pub d_speed: i32,       
    pub d_m_carry: i32,       
    pub d_longest: f64,     
}

#[derive(Clone, Copy)]
pub struct SolveInfo {
    pub d_dist: f64,
    pub d_carry: i32,
}

pub struct Solution {
    pub all_dist: f64,
    pub route: Vec<usize>,
}

pub struct DeliverySolver {
    pub adj_mat: AdjMat,
    pub pt_role: PointRole,

    pub config: Config,
    pub nearest_sender: HashMap<usize, usize>,
}

impl DeliverySolver {
    pub fn new(data: Vec<(i32, i32)>, s_ids: &Vec<usize>, r_ids: &Vec<usize>, conf: Config) -> Self {
        let points = Point::new_from(data);
        let adj_mat = AdjMat::new(points);
        let pt_role = PointRole::new(
            s_ids.to_owned(), 
            r_ids.to_owned()
        );

        let nearest_sender = DeliverySolver::init_near_map(&adj_mat, &pt_role);
        Self { adj_mat: adj_mat, pt_role: pt_role, config: conf, nearest_sender: nearest_sender }
    }

    fn init_near_map(adj_mat: &AdjMat, pt_role: &PointRole) -> HashMap<usize, usize> {
        let mut nearest_sender: HashMap<usize, usize> = HashMap::new();

        pt_role.recvers.iter().for_each(|&recver| {
            let recver_dist = &adj_mat
                .dist[recver];

            let dist_from_sender = pt_role
                .senders
                .iter()
                .map(|&sender| (
                    sender,
                    recver_dist[sender],
                )).collect::<HashMap<_, _>>();

            nearest_sender.insert(
                recver, 
                dist_from_sender
                    .into_iter()
                    .min_by_key(|(_, v)|
                        OrderedFloat(*v)
                    ).unwrap().0
            );
        });

        nearest_sender
    }

    fn can_be_chosen(&self, root: usize, pre_pt: usize, this_pt: usize, pri: Priority, info: SolveInfo) -> bool {
        let this_to_near = OrderedFloat(self.adj_mat.get_dist(this_pt, self.nearest_sender[&this_pt]));
        let this_to_pre = OrderedFloat(self.adj_mat.get_dist(this_pt, pre_pt));
        let this_to_root = OrderedFloat(self.adj_mat.get_dist(this_pt, root));
        let time_passed = ((this_to_pre + info.d_dist) / OrderedFloat(self.config.d_speed as f64)).ceil() as i32;

        let pre_to_root = OrderedFloat(self.adj_mat.get_dist(pre_pt, root));
        let dist_longest = OrderedFloat(self.config.d_longest);
        let dist_passed = OrderedFloat(info.d_dist);

        if this_to_pre + this_to_root > pre_to_root + this_to_near * 2. { false }   // not a good choice
        else if dist_passed + this_to_pre + this_to_root > dist_longest { false }   // exceed longest dist                                       
        else if info.d_carry >= self.config.d_m_carry { false }                     // no carry space left              
        else if time_passed > pri_to_num(pri) { false }                             // cannot satisfy priority    
        else { true }
    }

    pub fn get_sorted_orders(&self, orders: Vec<Order>) -> Vec<Order> {
        orders
            .into_iter()
            .sorted_by(|x, y| {
                let x_fit = OrderedFloat(
                    self
                        .adj_mat
                        .get_dist(
                            x.owned,
                            self.nearest_sender[&x.owned]
                        )
                );

                let y_fit = OrderedFloat(
                    self
                        .adj_mat
                        .get_dist(
                            y.owned,
                            self.nearest_sender[&y.owned]
                        )
                );

                x_fit.cmp(&y_fit)
            }).collect::<Vec<_>>()
    }

    pub fn build_mst_path(&self, root: usize, recvers: &HashSet<usize>) -> Vec<usize> {
        let mut tree = Tree::new_with_root(root);
        let mut left_recvs = recvers.clone();

        let root_edges = self.adj_mat.get_edges(root, &left_recvs);
        let mut edge_queue: BinaryHeap<Edge> = BinaryHeap::from(root_edges);

        while !left_recvs.is_empty() {
            let edge = edge_queue.pop().unwrap();
            if tree.data_exist(&edge.link_to) { continue; }
            tree.insert_node_by_data(&edge.linker, edge.link_to);
            
            left_recvs.remove(&edge.link_to);
            // edge_queue.retain(|e| e.link_to == edge.link_to);
            edge_queue.extend(self.adj_mat.get_edges(edge.link_to, &left_recvs));
        }

        tree.get_path()
    }

    pub fn prog_per_orders(&self, orders: &Vec<Order>) -> Vec<Solution> {  //directly solve the 1st pris' orders
        let c_orders = orders.clone();
        let sorted_orders = self.get_sorted_orders(c_orders);
        print_orders(&sorted_orders, "[Process queue] ");

        let mut recv_orders: HashMap<usize, Vec<usize>> = HashMap::new();
        sorted_orders
            .iter()
            .enumerate()
            .for_each(|(i, order)| {
                if !recv_orders.contains_key(&order.owned) {
                    recv_orders.insert(order.owned, vec![i]);
                } else {
                    recv_orders
                        .get_mut(&order.owned)
                        .unwrap()
                        .push(i);
                }
            });
        
        let mut handled: Vec<bool> = vec![false; sorted_orders.len()];
        let mut solutions: Vec<Solution> = Vec::with_capacity(sorted_orders.len());
 
        for (i, order) in sorted_orders.iter().enumerate() {
            if handled[i] { continue; }

            let mut info = SolveInfo { d_dist: 0., d_carry: 0 };
            let root = self.nearest_sender[&order.owned];
            let recvers = recv_orders
                .keys()
                .cloned()
                .collect::<HashSet<_>>();

            let mst_path = self.build_mst_path(
                root,
                &recvers
            );
            
            let mut real_path: VecDeque<usize> = VecDeque::new();
            for pt in mst_path.iter() {
                if real_path.is_empty() {
                    real_path.push_back(*pt);
                    continue;
                } 

                let last_pt = real_path.back().unwrap();
                if self.can_be_chosen(root, *last_pt, *pt, order.pri, info) {
                    if recv_orders[pt].is_empty() { continue; }

                    let to_hand = 
                        recv_orders[pt]
                        .len()
                        .min(
                            (self.config.d_m_carry - info.d_carry)
                                .abs()
                                as usize
                        );

                    (0..to_hand).for_each(|_| {
                        let order_i = recv_orders
                            .get_mut(pt)
                            .unwrap()
                            .pop()
                            .unwrap();

                        handled[order_i] = true;
                    });

                    info.d_carry += to_hand as i32;
                    info.d_dist += self.adj_mat.get_dist(*last_pt, *pt);
                    real_path.push_back(*pt);
                }
            }

            info.d_dist += self
                .adj_mat
                .get_dist(
                    *real_path
                        .back()
                        .unwrap(), 
                    root
                );
            real_path.push_back(root);

            solutions.push(Solution { 
                all_dist: info.d_dist, 
                route: real_path.into_iter().collect_vec()
            });
        }

        solutions
    }
}
