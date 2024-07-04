package org.example.entities;

public class Order {
    Priority priority;
    double createTime;
    double remainTime;
    Position position;

    public Order(Priority priority, double createTime, double remainTime, Position position) {
        this.priority = priority;
        this.createTime = createTime;
        this.remainTime = remainTime;
        this.position = position;
    }

    public Position getPosition() {
        return position;
    }

    public void setPosition(Position position) {
        this.position = position;
    }

    public Priority getPriority() {
        return priority;
    }

    public void setPriority(Priority priority) {
        this.priority = priority;
    }

    public double getCreateTime() {
        return createTime;
    }

    public void setCreateTime(double createTime) {
        this.createTime = createTime;
    }

    public double getRemainTime() {
        return remainTime;
    }

    public void setRemainTime(double remainTime) {
        this.remainTime = remainTime;
    }

    @Override
    public String toString() {
        return "Order{" +
                "priority=" + priority +
                ", createTime=" + createTime +
                ", remainTime=" + remainTime +
                ", position=" + position +
                "}\n";
    }
}
