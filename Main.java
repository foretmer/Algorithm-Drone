import java.util.*;

public class Main {
    static class Order{
        int normalNum;
        int spNum;
        int sspNum;
    }
    static class Path{
        List<Integer> path;
        int distance;
        Path(){
            path=new ArrayList<>();
        }
    }
    static int[][] adjacencyMatrix;
    static Order[][] orders;
    static Random random=new Random(42);
    public static void main(String[] args) {
        setAdjacencyMatrix();
        generateOrders();
        problemSolver();
    }
    static void setAdjacencyMatrix(){
        adjacencyMatrix=new int[][]{
                {0,0,0,0,0,3,2,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,4,0,0,0,2,2,0,3,0,0,0,5,0,0,0,0,0},
                {0,0,0,0,0,0,0,4,0,0,2,3,0,0,3,0,0,2,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,3,0,0,3,3,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
                {3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {2,0,0,0,0,0,0,0,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {3,0,4,0,0,0,0,0,0,1,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0},
                {0,4,0,0,0,0,3,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,4,1,0,0,0,0,0,1,0,0,0,0,4,0,0,0,0,0,0},
                {0,0,2,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
                {0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,3,0,0,0,0,0,0,0},
                {0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0},
                {0,2,0,3,0,0,0,0,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3},
                {0,0,3,0,0,0,0,3,0,0,0,0,0,0,0,0,0,4,2,0,0,0,0,0,0},
                {0,3,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0},
                {0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},
                {0,0,2,0,0,0,0,0,0,0,0,3,0,0,4,0,0,0,0,0,0,0,2,0,0},
                {0,0,0,0,0,0,0,0,0,4,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0},
                {0,5,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0},
                {0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0},
                {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0},
                {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,0,1,0},
                {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1},
                {0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,1,0}
        };
        printAdjacencyMatrix();
    }
    static void printAdjacencyMatrix(){
        System.out.println("邻接矩阵:");
        for(int i=0;i<25;i++){
            for(int j=0;j<25;j++) {
                if(j!=0)System.out.print(" ");
                System.out.print(adjacencyMatrix[i][j]);
            }
            System.out.println();
        }
    }
    static void generateOrders(){
        orders=new Order[48][20];
        for(int i=0;i<48;i++){
            for(int j=0;j<20;j++){
                orders[i][j]=new Order();
            }
            int orderCount=random.nextInt(51);
            while(orderCount!=0){
                orderCount--;
                int index=random.nextInt(20);
                int p=random.nextInt(10);
                if(p<5){
                    orders[i][index].normalNum++;
                }else if(p<8){
                    orders[i][index].spNum++;
                }else{
                    orders[i][index].spNum++;
                }
            }
        }
        printOrders();
    }
    static void printOrders(){
        System.out.println("订单：");
        for(int i=0;i<48;i++){
            for(int j=0;j<20;j++){
                System.out.print("("+(j+6)+" "+orders[i][j].normalNum+" "+orders[i][j].spNum+" "+orders[i][j].sspNum+")");
                if(j!=19) System.out.print(",");
                else System.out.println();
            }
        }
    }
    static void problemSolver(){
        //生成每个配送中心的距离小于等于20的全部回环路径
        Map<Integer,List<Path>> loopMap=new HashMap<>();
        for(int i=1;i<=5;i++){
            int index=i-1;
            List<Path> paths=new ArrayList<>();
            Queue<Path> q=new LinkedList<>();
            Path pth=new Path();
            pth.path.add(index);
            q.add(pth);
            while(q.size()!=0){
                pth=q.poll();
                if(pth.distance>20) continue;
                if(pth.distance!=0&&pth.path.getLast()==index) {
                    pth.path.removeLast();
                    pth.path.removeFirst();
                    paths.add(pth);
                    continue;
                }
                for(int j=5;j<25;j++){
                    if(adjacencyMatrix[pth.path.getLast()][j]!=0){
                        if(pth.path.size()>=2&&pth.path.get(pth.path.size()-2)==j) continue;
                        Path t=new Path();
                        t.distance=pth.distance;
                        t.path=new ArrayList<>(pth.path);
                        t.distance+=adjacencyMatrix[pth.path.getLast()][j];
                        t.path.add(j);
                        q.add(t);
                    }
                }
                if(adjacencyMatrix[pth.path.getLast()][index]!=0){
                    Path t=new Path();
                    t.distance=pth.distance;
                    t.path=new ArrayList<>(pth.path);
                    t.distance+=adjacencyMatrix[pth.path.getLast()][index];
                    t.path.add(index);
                    q.add(t);
                }
            }
            Collections.sort(paths, Comparator.comparingInt((Path a) -> a.distance));
            List<Path> removeDuplicatesPaths=new ArrayList<>();
            for(int k=0;k<paths.size();k++){
                boolean mark=true;
                for(int w=0;w<removeDuplicatesPaths.size();w++){
                    if(removeDuplicatesPaths.get(w).distance==paths.get(k).distance
                            &&removeDuplicatesPaths.get(w).path.size()==paths.get(k).path.size()){
                        boolean flag=true;
                        for(int x=0;x<paths.get(k).path.size();x++){
                            if(removeDuplicatesPaths.get(w).path.get(removeDuplicatesPaths.get(w).path.size()-x-1)
                                    !=paths.get(k).path.get(x)) {
                                flag=false;
                                break;
                            }
                        }
                        if(flag) {
                            mark=false;
                            break;
                        }
                    }
                }
                if(mark) removeDuplicatesPaths.add(paths.get(k));
            }
            loopMap.put(i,removeDuplicatesPaths);
        }
        System.out.println("每个配送中心的距离小于等于20的全部回环路径:");
        for(int i:loopMap.keySet()){
            System.out.println("配送中心"+i+"：");
            List<Path> paths=loopMap.get(i);
            for(Path path:paths){
                System.out.print("总距离:"+path.distance+",路径:");
                for(int j=0;j<path.path.size();j++){
                    System.out.print(path.path.get(j)+1);
                    if(j!=path.path.size()-1) System.out.print('-');
                    else System.out.println();
                }
            }
        }
        //生成每个卸货点到配送中心的最短路径
        Map<Integer,Path> minimalDistanceMap=new HashMap<>();
        for(int i=5;i<25;i++){
            PriorityQueue<Path> q=new PriorityQueue<>(Comparator.comparingInt((Path x) -> x.distance));
            Path pth=new Path();
            pth.path.add(i);
            q.add(pth);
            while(q.size()!=0){
                pth=q.poll();
                if(pth.distance>10) continue;
                if(pth.path.getLast()<5) {
                    minimalDistanceMap.put(i,pth);
                    break;
                }
                for(int j=0;j<25;j++){
                    if(adjacencyMatrix[pth.path.getLast()][j]!=0){
                        if(pth.path.size()>=2&&pth.path.get(pth.path.size()-2)==j) continue;
                        Path t=new Path();
                        t.distance=pth.distance;
                        t.path=new ArrayList<>(pth.path);
                        t.distance+=adjacencyMatrix[pth.path.getLast()][j];
                        t.path.add(j);
                        q.add(t);
                    }
                }
            }
        }
        System.out.println("每个卸货点到配送中心的最短路径:");
        for(int i=5;i<25;i++){
            Path ph=minimalDistanceMap.get(i);
            System.out.print("卸货点:"+(i+1)+",距离:"+ph.distance+",路径:");
            for(int k=0;k<ph.path.size();k++){
                System.out.print(ph.path.get(k)+1);
                if(k!=ph.path.size()-1) System.out.print('-');
                else System.out.println();
            }
        }
        //计算每个卸货点到达配送中心小于等于19的全部路径，按距离分组
        Map<Integer,Map<Integer,List<Path>>> returnMap=new HashMap<>();
        for(int i=5;i<25;i++){
            returnMap.put(i,new HashMap<>());
            Queue<Path> q=new LinkedList<>();
            Path pth=new Path();
            pth.path.add(i);
            q.add(pth);
            while(q.size()!=0){
                pth=q.poll();
                if(pth.distance>19) continue;
                if(pth.path.getLast()<5) {
                    if (returnMap.get(i).get(pth.distance) ==null){
                        List<Path> li=new ArrayList<>();
                        li.add(pth);
                        returnMap.get(i).put(pth.distance,new ArrayList<>(li));
                    }else returnMap.get(i).get(pth.distance).add(pth);
                    continue;
                }
                for(int j=0;j<25;j++){
                    if(adjacencyMatrix[pth.path.getLast()][j]!=0){
                        if(pth.path.size()>=2&&pth.path.get(pth.path.size()-2)==j) continue;
                        Path t=new Path();
                        t.distance=pth.distance;
                        t.path=new ArrayList<>(pth.path);
                        t.distance+=adjacencyMatrix[pth.path.getLast()][j];
                        t.path.add(j);
                        q.add(t);
                    }
                }
            }
        }
        System.out.println("每个卸货点到配送中心的小于等于19的路径:");
        for(int i=5;i<25;i++){
            System.out.println("卸货点"+(i+1)+":");
            Map<Integer,List<Path>> singleLocation=returnMap.get(i);
            for(int k:singleLocation.keySet()){
                System.out.print("距离为"+k+"的路径:");
                for(Path p:singleLocation.get(k)){
                    for(int j=0;j<p.path.size();j++){
                        System.out.print(p.path.get(j)+1);
                        if(j!=p.path.size()-1) System.out.print('-');
                        else System.out.print(',');
                    }
                }
                System.out.println();
            }
        }
        PriorityQueue<Integer> waitingOrder[]=new PriorityQueue[20];
        //订单动态分配配送中心，放入等待池批量响应
        int nowtime=0;
        for(int i=0;i<48;i++){
            System.out.println("时间："+(30*i));
            for(int j=0;j<20;j++){
                waitingOrder[j]=new PriorityQueue<>();
                int normalNum=orders[i][j].normalNum;
                int spNum=orders[i][j].spNum;
                int sspNum=orders[i][j].sspNum;
                for(int k=0;k<normalNum;k++) waitingOrder[j].add(nowtime+30);
                for(int k=0;k<spNum;k++) waitingOrder[j].add(nowtime+90);
                for(int k=0;k<sspNum;k++) waitingOrder[j].add(nowtime+180);
            }
            System.out.println("    批量响应阶段：");
            for(int j=0;j<5;j++){
                boolean mark=false;
                List<Path> deliveryCenterPaths=loopMap.get(j+1);
                for(Path p:deliveryCenterPaths){
                    int cnt;
                    do{
                        cnt=0;
                        List<Integer> opNum=new ArrayList<>();
                        for(int item:p.path){
                            int min=Math.min(10-cnt,waitingOrder[item-5].size());
                            cnt+=min;
                            opNum.add(min);
                            if(cnt==10) {
                                if(!mark){
                                    mark=true;
                                    System.out.println("        配送中心:"+(j+1));
                                }
                                System.out.println("          检测到无人机满载订单进行响应：");
                                System.out.print("          ");
                                for(int x=0;x<opNum.size();x++){
                                    int count=opNum.get(x);
                                    int location=p.path.get(x);
                                    System.out.print("卸货点："+(location+1)+"-响应订单数："+count);
                                    if(x!=opNum.size()-1) System.out.print(",");
                                    else System.out.println();
                                    while(count!=0){
                                        waitingOrder[location-5].poll();
                                        count--;
                                    }
                                }
                                break;
                            }
                        }
                    }while(cnt==10);
                }
            }
            System.out.println("    最大等待时间订单响应阶段：");
            for(int y=0;y<20;y++){
                int cnt=0;
                while(waitingOrder[y].size()>0&&waitingOrder[y].peek()==nowtime+30){
                    waitingOrder[y].poll();
                    cnt++;
                }
                Path p= minimalDistanceMap.get(y+5);
                while(cnt>10){
                    System.out.println("          检测到最大等待时间订单进行响应：");
                    System.out.print("          ");
                    for(int z=0;z<p.path.size();z++){
                        int count=p.path.get(z)==y+5?10:0;
                        int location=p.path.get(z);
                        System.out.print("卸货点："+(location+1)+"-响应订单数："+count);
                        if(z!=p.path.size()-1) System.out.print(",");
                        else System.out.println();
                    }
                    cnt-=10;
                }
                if(cnt>0){
                    int remains=10-cnt;
                    System.out.print("          检测到最大等待时间订单进行响应：");
                    for(int z=p.path.size()-2;z>=1;z--){
                        int min=Math.min(remains,waitingOrder[p.path.get(z)-5].size());
                        int count=min;
                        int location=p.path.get(z);
                        System.out.print("卸货点："+(location+1)+"-响应订单数："+count);
                        remains-=min;
                        while(min!=0){
                            waitingOrder[p.path.get(z)-5].poll();
                            min--;
                        }
                    }
                    System.out.println("卸货点："+(y+5)+"-响应订单数："+cnt);
                }
            }
            nowtime+=30;
        }
    }
}