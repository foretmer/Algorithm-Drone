package org.example.simulations;

import org.example.entities.*;
import org.example.utils.Util;

import java.util.ArrayList;
import java.util.Random;

public class OrderList {
    public ArrayList<Order> orders;

    public OrderList() {
    }

    public OrderList(ArrayList<Order> orders) {
        this.orders = orders;
    }

    public void newOrders(DischargePoint[] dischargePoints, int m, double currentTime) {
        orders = new ArrayList<>();
        for (DischargePoint dischargePoint : dischargePoints) {
            Random random = new Random();
            int num = random.nextInt(m + 1);
            for (int i = 0; i < num; i++) {
                Priority priority = Util.randomPriority();
                double remainTime = 0;
                if (priority == Priority.LOW) {
                    remainTime += 180;
                }
                if (priority == Priority.MEDIUM) {
                    remainTime += 90;
                }
                if (priority == Priority.HIGH) {
                    remainTime += 30;
                }
                Position position = dischargePoint.getPosition();
                orders.add(new Order(priority, currentTime, remainTime, position));
            }
        }
    }

    public void orderDistribution(DistributionCenter[] distributionCenters) {
        for (Order order : orders) {
            double minDist = Double.MAX_VALUE;
            int flag = 0;
            for (int j = 0; j < distributionCenters.length; j++) {
                double d = Util.distance(order.getPosition(), distributionCenters[j].getPosition());
                if (d < minDist) {
                    minDist = d;
                    flag = j;
                }
            }
            distributionCenters[flag].orders.add(order);
        }
        orders.clear();
    }
}
