package org.example.utils;

import org.example.entities.Position;
import org.example.entities.Priority;

import java.util.ArrayList;
import java.util.Random;

public class Util {
    public static Random random = new Random();

    public static double distance(Position x, Position y) {
        return Math.sqrt(Math.pow(x.getX() - y.getX(), 2) + Math.pow(x.getY() - y.getY(), 2));
    }

    public static Priority randomPriority() {
        return Priority.values()[random.nextInt(Priority.values().length)];
    }

    public static double pathLength(ArrayList<Position> path) {
        double totalLength = 0.0;
        for (int i = 0; i < path.size(); i++) {
            Position current = path.get(i);
            Position next = path.get((i + 1) % path.size());
            totalLength += distance(current, next);
        }
        return totalLength;
    }

    public static double pathTime(ArrayList<Position> path, double speed) {
        double totalLength = pathLength(path);
        return totalLength / speed;
    }
}
