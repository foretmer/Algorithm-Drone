package org.example.entities;

import org.example.utils.Util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class DistributionCenter {
    Position position;
    public ArrayList<Order> orders;
    public ArrayList<Drone> drones;

    public DistributionCenter(Position position, ArrayList<Order> orders, ArrayList<Drone> drones) {
        this.position = position;
        this.orders = orders;
        this.drones = drones;
    }

    public DistributionCenter(Position position) {
        this.position = position;
        this.orders = new ArrayList<>();
        this.drones = new ArrayList<>();
    }

    public Position getPosition() {
        return position;
    }

    public void setPosition(Position position) {
        this.position = position;
    }

    public ArrayList<Order> getOrders() {
        return orders;
    }

    public void setOrders(ArrayList<Order> orders) {
        this.orders = orders;
    }

    public ArrayList<Drone> getDrones() {
        return drones;
    }

    public void setDrones(ArrayList<Drone> drones) {
        this.drones = drones;
    }

    @Override
    public String toString() {
        return position.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof DistributionCenter other) {
            return this.position.equals(other.position);
        }
        return false;
    }

    @Override
    public int hashCode() {
        try {
            return position.hashCode();
        } catch (Exception e) {
            Random random = new Random();
            return random.nextInt(100);
        }
    }

    public double droneSchedule(int n, double maxDis, double s, boolean flag) {
        System.out.println("***************************************");
        System.out.println("当前该中心订单总数：" + orders.size());
        ArrayList<Order> ordersPlan = new ArrayList<>();
        if (flag) {
            ordersPlan.addAll(orders);
            orders.clear();
        } else {
            Random random = new Random();
            ArrayList<Integer> removeOrders = new ArrayList<>();
            for (int i = 0; i < orders.size(); i++) {
                if (orders.get(i).remainTime <= 30) {
                    ordersPlan.add(orders.get(i));
                    removeOrders.add(i);
                } else {
                    if (random.nextInt(2) == 1) {
                        ordersPlan.add(orders.get(i));
                        removeOrders.add(i);
                    }
                }
            }
            removeOrders.sort(Collections.reverseOrder());
            for (int i : removeOrders) {
                orders.remove(i);
            }
        }
        System.out.println("本次配送订单个数：" + ordersPlan.size());
        while (!ordersPlan.isEmpty()) {
            Drone drone = new Drone();
            drone.path.add(position);
            ArrayList<Integer> removeOrders = new ArrayList<>();
            for (int i = 0; i < ordersPlan.size(); i++) {
                drone.path.add(ordersPlan.get(i).getPosition());
                if (drone.currentLoad < n && Util.pathLength(drone.path) <= maxDis && Util.pathTime(drone.path, s) <= ordersPlan.get(i).remainTime) {
                    drone.currentLoad++;
                    removeOrders.add(i);
                    continue;
                }
                drone.path.removeLast();
            }
            removeOrders.sort(Collections.reverseOrder());
            for (int i : removeOrders) {
                ordersPlan.remove(i);
            }
            removeOrders.clear();
            drones.add(drone);
        }
        System.out.println("本配送中心本次出动无人机：" + drones.size());
        System.out.println("***************************************");

        double totalDistance = 0;
        for (Drone drone : drones) {
            totalDistance += Util.pathLength(drone.getPath());
        }
        return totalDistance;
    }
}
