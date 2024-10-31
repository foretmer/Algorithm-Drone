package org.example.entities;

import java.util.ArrayList;

public class Drone {
    public ArrayList<Position> path;
    int currentLoad;

    public Drone() {
        this.path = new ArrayList<>();
        this.currentLoad = 0;
    }

    public Drone(ArrayList<Position> path, int currentLoad) {
        this.path = path;
        this.currentLoad = currentLoad;
    }

    public ArrayList<Position> getPath() {
        return path;
    }

    public void setPath(ArrayList<Position> path) {
        this.path = path;
    }

    public int getCurrentLoad() {
        return currentLoad;
    }

    public void setCurrentLoad(int currentLoad) {
        this.currentLoad = currentLoad;
    }

    @Override
    public String toString() {
        return "Drone{" +
                "path=" + path +
                ", currentLoad=" + currentLoad +
                '}';
    }
}
