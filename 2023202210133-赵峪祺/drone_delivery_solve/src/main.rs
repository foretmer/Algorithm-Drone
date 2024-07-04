use std::error::Error;

use crate::solver::{order::{OrderGener, Scheduler, print_orders}, solve::{Config, DeliverySolver}, order::GenWay};

mod solver;

fn main() -> Result<(), Box<dyn Error>> {
    const TOTAL_TIME: i32 = 180;        // total time, I set it to 180(min)
    const GEN_DURATION: i32 = 30;       // duration between gen, I set it to 30(min)
    const GEN_ORDERS: usize = 6;        // gen orders' number, I set it to 3

    let point_datas = vec![                 // point datas
        (1, 1), (5, 3), (2, 6), 
        (7, 2), (4, 8), (8, 4), 
        (3, 7), (6, 5), (9, 9),
    ];
    let s_ids = vec![1, 4, 7];                  // senders
    let r_ids = vec![0, 2, 3, 5, 6, 8];         // receivers

    let solver = DeliverySolver::new(
        point_datas, 
        &s_ids, 
        &r_ids, 
        Config {
            d_speed: 1,         // km per min: 1
            d_longest: 20.,     // max carry, I set it to 3
            d_m_carry: 3,       // longest fly dist: 20
        }
    );

    solver.adj_mat.print_mat();
    let mut scheduler = Scheduler::new();
    let mut order_gener = OrderGener::new(&s_ids, &r_ids);
    
    let total_cost = (0..(TOTAL_TIME / GEN_DURATION)).map(|i| {
        println!("======");
        println!("Time slice {}", i + 1);

        let orders = order_gener.gen(GEN_ORDERS, GenWay::Every);
        print_orders(&orders, "[New orders] ");
        
        scheduler.parse_orders(orders);
        if i == TOTAL_TIME / GEN_DURATION - 1 {
            scheduler.fin_round();
        }

        let high_orders = scheduler.get_high_list();
        print_orders(&high_orders, "[High-pri orders] ");
        let sols = solver.prog_per_orders(high_orders);

        println!("[Routes and costs]");
        let summed = sols.iter().map(|sol| {
            println!(
                "\tMin cost: {:.4} \tRoute: {:?}", 
                sol.all_dist, 
                sol.route
            );
            sol.all_dist
        }).sum::<f64>();

        println!("[Total min cost] {:.4}", summed);
        scheduler.sift_up();
        summed
    }).sum::<f64>();

    println!("======");
    println!("[Overall cost] {:.4}", total_cost);
    Ok(())
}