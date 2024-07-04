use super::graph::PointRole;
use rand::{rngs::ThreadRng, seq::IteratorRandom, thread_rng};

#[derive(Clone, Copy)]
pub enum Priority {
    High,
    Mid,
    Low,
}

pub enum GenWay {
    Every,
    OnlyHigh,
    OnlyMid,
    OnlyLow,
}

#[derive(Clone, Copy)]
pub struct Order {
    pub pri: Priority,       // unit: minute, 30: 1st | 90: 2nd | 180: 3rd
    pub owned: usize,          // the order hold by which point
}

pub struct OrderGener {
    pub pt_role: PointRole,
    pub rng: ThreadRng,
}

pub struct Scheduler {
    pub high_list: Vec<Order>,
    pub mid_list: Vec<Order>,
    pub low_list: Vec<Order>,
}

pub fn pri_to_num(pri: Priority) -> i32 {
    match pri {
        Priority::High => 30,
        Priority::Mid => 90,
        Priority::Low => 180,
    }
}

pub fn print_orders(orders: &Vec<Order>, s: &str) {
    print!("{}", s);
    orders.iter().for_each(|order| {
        print!("{} ", order.owned);
        match order.pri {
            Priority::High => print!("High "),
            Priority::Mid => print!("Mid "),
            Priority::Low => print!("Low "),
        }
    });
    println!();
}

impl OrderGener {
    pub fn new(s_ids: &Vec<usize>, r_ids: &Vec<usize>) -> Self {
        Self { 
            pt_role: PointRole::new(
                s_ids.to_owned(), 
                r_ids.to_owned(),
            ), 
            rng: thread_rng() 
        }
    }

    pub fn gen(&mut self, n: usize, way: GenWay) -> Vec<Order> {
        let chosen = self
            .pt_role
            .recvers
            .clone()
            .into_iter()
            .choose_multiple(&mut self.rng, n);
        
        let enums = [
            Priority::High,
            Priority::Mid,
            Priority::Low,
        ];

        match way {
            GenWay::Every => chosen.into_iter().map(|it| {
                Order {
                    owned: it,
                    pri: enums
                        .iter()
                        .choose(&mut self.rng)
                        .unwrap()
                        .to_owned()
                }
            }).collect::<Vec<_>>(),

            GenWay::OnlyHigh => chosen.into_iter().map(|it| {
                Order {
                    owned: it,
                    pri: Priority::High,
                }
            }).collect::<Vec<_>>(),

            GenWay::OnlyMid => chosen.into_iter().map(|it| {
                Order {
                    owned: it,
                    pri: Priority::Mid,
                }
            }).collect::<Vec<_>>(),

            GenWay::OnlyLow => chosen.into_iter().map(|it| {
                Order {
                    owned: it,
                    pri: Priority::Low,
                }
            }).collect::<Vec<_>>(),
        }
    }
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            high_list: Vec::new(),
            mid_list: Vec::new(),
            low_list: Vec::new(),
        }
    }

    pub fn sift_up(&mut self) {
        self.high_list.clear();
        self.high_list
            .extend(
                self.mid_list
                    .clone()
                    .iter()
                    .map(|it| 
                        Order {
                            pri: Priority::High,
                            ..*it
                        }
                    )
            );

        self.mid_list.clear();
        self.mid_list
            .extend(
                self.low_list
                    .iter()
                    .map(|it| 
                        Order {
                            pri: Priority::Mid,
                            ..*it
                        }
                    )
            );

        self.low_list.clear();
    }

    pub fn fin_round(&mut self) {
        self.high_list
            .extend(
                self.mid_list
                    .clone()
                    .iter()
                    .map(|it| 
                        Order {
                            pri: Priority::High,
                            ..*it
                        }
                    )
            );

        self.high_list
            .extend(
                self.low_list
                    .iter()
                    .map(|it| 
                        Order {
                            pri: Priority::High,
                            ..*it
                        }
                    )
            );
    }

    pub fn parse_orders(&mut self, orders: Vec<Order>) {
        orders.iter().for_each(|order| {
            match order.pri {
                Priority::High => self.high_list.push(*order),
                Priority::Mid => self.mid_list.push(*order),
                Priority::Low => self.low_list.push(*order),
            }
        })
    }

    pub fn get_high_list(&self) -> &Vec<Order> {
        &self.high_list
    }
}

