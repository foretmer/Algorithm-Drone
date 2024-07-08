use std::collections::HashSet;
use ordered_float::OrderedFloat;

pub struct Point {
    pub x: i32,
    pub y: i32,
}

pub struct AdjMat {
    pub dist: Vec<Vec<f64>>,
    pub points: Vec<Point>,
}

#[derive(Clone)]
pub struct PointRole {
    pub senders: HashSet<usize>,
    pub recvers: HashSet<usize>,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Edge {
    pub linker: usize,
    pub link_to: usize,
    pub dist: OrderedFloat<f64>,
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist.cmp(&self.dist)
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x: x, y: y }
    }

    pub fn new_from(data: Vec<(i32, i32)>) -> Vec<Self> {
        data.iter().map(|it|
            Point::new(it.0, it.1)
        ).collect::<Vec<_>>()
    }
}

impl std::ops::Sub for &Point {
    type Output = f64;

    fn sub(self, rhs: Self) -> Self::Output {
        (((rhs.x - self.x).pow(2) + (rhs.y - self.y).pow(2)) as f64).sqrt()
    }
}

impl AdjMat {
    pub fn new(points: Vec<Point>) -> Self {
        Self { dist: AdjMat::get_all_dist(&points), points: points }
    }

    fn get_all_dist(points: &Vec<Point>) -> Vec<Vec<f64>> {
        points.iter().map(|pt_out| 
            points.iter().map(|pt_in|
                pt_out - pt_in
            ).collect::<Vec<_>>()
        ).collect::<Vec<_>>()
    }

    pub fn get_dist(&self, pt_a: usize, pt_b: usize) -> f64 {
        self.dist[pt_a][pt_b]
    }

    pub fn get_edges(&self, pt: usize, other_pts: &HashSet<usize>) -> Vec<Edge> {
        let mut edges: Vec<Edge> = Vec::new();

        self.dist[pt].iter().enumerate().for_each(|(p, d)| {
            if p != pt && other_pts.contains(&p) {
                edges.push(Edge { 
                    linker: pt, 
                    link_to: p, 
                    dist: OrderedFloat(*d)
                })
            }
        });

        edges
    }

    pub fn print_mat(&self) {
        println!("======\nAdjMat:");
        self.dist.iter().for_each(|its| {
            its.iter().for_each(|it| {
                print!("{:.4} ", it);
            });
            println!();
        });
    }
}

impl PointRole {
    pub fn new(s_ids: Vec<usize>, r_ids: Vec<usize>) -> Self {
        Self { 
            senders: s_ids.into_iter().collect::<HashSet<_>>(), 
            recvers: r_ids.into_iter().collect::<HashSet<_>>()
        }
    }
}

