package org.example.entities;

import java.util.Random;

public class DischargePoint {
    Position position;

    public DischargePoint(Position position) {
        this.position = position;
    }

    public Position getPosition() {
        return position;
    }

    public void setPosition(Position position) {
        this.position = position;
    }

    @Override
    public String toString() {
        return position.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof DischargePoint other) {
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
}
