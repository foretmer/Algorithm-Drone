package org.example.simulations;

import org.example.entities.DischargePoint;
import org.example.entities.DistributionCenter;
import org.example.entities.Position;
import org.example.utils.Util;

import java.util.*;

public class Init {
    DistributionCenter[] distributionCenters;
    DischargePoint[] dischargePoints;

    public Init() {
    }

    public Init(DistributionCenter[] distributionCenters, DischargePoint[] dischargePoints) {
        this.distributionCenters = distributionCenters;
        this.dischargePoints = dischargePoints;
    }

    public DistributionCenter[] getDistributionCenters() {
        return distributionCenters;
    }

    public void setDistributionCenters(DistributionCenter[] distributionCenters) {
        this.distributionCenters = distributionCenters;
    }

    public DischargePoint[] getDischargePoints() {
        return dischargePoints;
    }

    public void setDischargePoints(DischargePoint[] dischargePoints) {
        this.dischargePoints = dischargePoints;
    }

    @Override
    public String toString() {
        return "Init{" + "distributionCenters=" + Arrays.toString(distributionCenters) + ", dischargePoints=" + Arrays.toString(dischargePoints) + '}';
    }

    public void createMap(int scaleX, int scaleY, int j, int k) {
        Random random = new Random();
        Set<DistributionCenter> distributionCenterSet = new HashSet<>();
        Set<DischargePoint> dischargePointSet = new HashSet<>();
        while (distributionCenterSet.size() < j) {
            int x = random.nextInt(scaleX);
            int y = random.nextInt(scaleY);
            if (distributionCenterSet.isEmpty()) {
                distributionCenterSet.add(new DistributionCenter(new Position(x, y)));
            }
            boolean flag = true;
            for (DistributionCenter center : distributionCenterSet) {
                if (Util.distance(center.getPosition(), new Position(x, y)) < 8) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                distributionCenterSet.add(new DistributionCenter(new Position(x, y)));
            }
        }
        while (dischargePointSet.size() < k) {
            int x = random.nextInt(scaleX);
            int y = random.nextInt(scaleY);
            for (DistributionCenter center : distributionCenterSet) {
                if (Util.distance(center.getPosition(), new Position(x, y)) < 10) {
                    dischargePointSet.add(new DischargePoint(new Position(x, y)));
                    break;
                }
            }
        }
        this.distributionCenters = new DistributionCenter[j];
        Iterator<DistributionCenter> iter1 = distributionCenterSet.iterator();
        for (int i = 0; i < j; i++) {
            this.distributionCenters[i] = iter1.next();
        }
        this.dischargePoints = new DischargePoint[k];
        Iterator<DischargePoint> iter2 = dischargePointSet.iterator();
        for (int i = 0; i < k; i++) {
            this.dischargePoints[i] = iter2.next();
        }
    }

    public void loadMap(int type) {
        int[][] pos1 = {{7, 10}, {8, 2}, {25, 16}};
        int[][] pos2 = {{28, 19}, {20, 14}, {19, 18}, {27, 16}, {13, 15}, {28, 10}, {13, 10},
                {26, 8}, {5, 15}, {3, 5}, {10, 19}, {16, 16}, {23, 10}, {13, 3}, {11, 4}};
        int[][] pos3 = {{6, 34}, {48, 23}, {53, 0}, {15, 9}, {43, 13}, {23, 37}, {58, 19}, {6, 5}, {32, 4}, {18, 18}};
        int[][] pos4 = {{52, 15}, {49, 28}, {46, 21}, {28, 6}, {25, 11}, {40, 17}, {24, 30}, {30, 4}, {48, 30}, {35, 3},
                {49, 4}, {15, 5}, {13, 37}, {23, 19}, {19, 36}, {4, 39}, {32, 1}, {15, 27}, {14, 37}, {7, 13},
                {46, 24}, {39, 4}, {16, 26}, {54, 11}, {23, 15}, {33, 6}, {30, 39}, {8, 37}, {17, 20}, {23, 28},
                {54, 11}, {16, 3}, {11, 32}, {13, 31}, {1, 2}, {15, 35}, {58, 14}, {19, 38}, {3, 5}, {44, 3}};

        switch (type) {
            case 0: {
                this.distributionCenters = new DistributionCenter[3];
                for (int i = 0; i < 3; i++) {
                    this.distributionCenters[i] = new DistributionCenter(new Position(pos1[i][0], pos1[i][1]));
                }
                this.dischargePoints = new DischargePoint[15];
                for (int i = 0; i < 15; i++) {
                    this.dischargePoints[i] = new DischargePoint(new Position(pos2[i][0], pos2[i][1]));
                }
                break;
            }
            case 1: {
                this.distributionCenters = new DistributionCenter[10];
                for (int i = 0; i < 10; i++) {
                    this.distributionCenters[i] = new DistributionCenter(new Position(pos3[i][0], pos3[i][1]));
                }
                this.dischargePoints = new DischargePoint[40];
                for (int i = 0; i < 40; i++) {
                    this.dischargePoints[i] =  new DischargePoint(new Position(pos4[i][0], pos4[i][1]));
                }
                break;
            }
            default: {
                System.out.println("Choose Correct Map Type.");
            }
        }
    }
}
