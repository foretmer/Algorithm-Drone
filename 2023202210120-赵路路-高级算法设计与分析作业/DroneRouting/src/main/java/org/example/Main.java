package org.example;

import org.example.entities.DistributionCenter;
import org.example.entities.Drone;
import org.example.entities.Position;
import org.example.entities.Priority;
import org.example.simulations.Init;
import org.example.simulations.OrderList;
import org.example.utils.Util;

import java.util.ArrayList;
import java.util.Arrays;

public class Main {
    public static int currentTime = 0;
    public static int maxIter = 48;

    public static void main(String[] args) {
        Init init = new Init();
        init.loadMap(1);

//        System.out.println(Arrays.toString(init.getDistributionCenters()));
//        System.out.println(Arrays.toString(init.getDischargePoints()));

//        for (int i = 0; i < 20; i++) {
//            System.out.println(Util.randomPriority());
//        }
//        System.out.println(Priority.HIGH);
//        System.out.println(Priority.MEDIUM);
//        System.out.println(Priority.LOW);

//        OrderList orderList = new OrderList();
//        orderList.newOrders(init.getDischargePoints(), 10, 0);
//        System.out.println(orderList.orders);
//
//        System.out.println();
//        orderList.orderDistribution(init.getDistributionCenters());
//        for (DistributionCenter distributionCenter : init.getDistributionCenters()) {
//            System.out.println(distributionCenter.orders);
//        }
//        System.out.println(orderList.orders);

//        ArrayList<Position> path = new ArrayList<>();
//        path.add(new Position(6, 34));
//        path.add(new Position(13, 37));
//        path.add(new Position(14, 37));
//        path.add(new Position(11, 32));
//        double length = Util.pathLength(path);
//        System.out.println("Path length: " + length);

//        System.out.println();
//        for (DistributionCenter distributionCenter : init.getDistributionCenters()) {
//            distributionCenter.droneSchedule(5, 20, 1);
//            for (Drone Drone : distributionCenter.getDrones()) {
//                System.out.println(Drone.getPath());
//            }
//        }

        OrderList orderList = new OrderList();
        for (int i = 1; i <= maxIter; i++) {
            System.out.println();
            System.out.println("Iteration #" + i);
            orderList.newOrders(init.getDischargePoints(), 10, currentTime);
            orderList.orderDistribution(init.getDistributionCenters());
            if (i == 48) {
                double d = 0;
                for (DistributionCenter distributionCenter : init.getDistributionCenters()) {
                    d += distributionCenter.droneSchedule(5, 20, 1, true);
                }
                System.out.println("本次航行总距离：" + d);
            } else {
                double d = 0;
                for (DistributionCenter distributionCenter : init.getDistributionCenters()) {
                    d += distributionCenter.droneSchedule(5, 20, 1, false);
                    for (Drone drone : distributionCenter.drones) {
                        System.out.println(drone.getPath());
                    }
                }
                System.out.println("本次航行总距离：" + d);
            }
            for (DistributionCenter distributionCenter : init.getDistributionCenters()) {
                distributionCenter.drones.clear();
            }
            currentTime += 30;
        }
    }
}
