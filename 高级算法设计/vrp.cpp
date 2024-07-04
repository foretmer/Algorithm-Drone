#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <ctime>
#include <queue>
using namespace std;

//定义卸货点
struct node {
    int num, x, y;
};

// 定义订单结构体
struct Order {
    int id;            // 订单ID
    int priority;      // 订单优先级(1-紧急,2-较紧急,3-一般)
    int nums;          // 卸货点订单数目
    int node_id;       // 卸货点位置
    int delay_cur;     // 延迟次数
    int delay_total;   // 总延迟次数
};

//定义个体
struct solution {
    vector<int> path;
    double sum, F, P;
    int gen;
};

int m = 5; // 卸货点最多生成订单数
int n = 10; // 单个无人机单次最大配送数
std::vector<Order> orders;
std::vector<Order> orders_last;

int curGen = 1; //当前代数
const int maxGen = 10000; //最大代数
const int start = 0;    //起点 (7,7)
int packNum = 20;    //种群容量
double mutationP = 0.04;   //变异概率
double crossP = 0.7;     //交叉概率
double curCost = 0; //当前记录的最优解，用于传代统计
int Hayflick = 6;   //最大传代次数
vector<node> city;  //订单需配送的卸货点集合
vector<int> num;    //订单需配送的卸货点序列, 用于shuffle
vector<solution> pack;  //种群
unordered_map<int, unordered_map<int, double> > dis;    //记录订单需配送的卸货点两两之间的距离
queue<double> record;   //传代统计队列

//计算两个卸货点之间的距离
double distanceBetween(node &a, node &b) {
    int x = a.x - b.x;
    int y = a.y - b.y;
    return sqrt(x * x + y * y);
}

//计算个体的解
double sum(vector<int> &v) {
    double sum = 0;
    int cur_load = 0;
    double cur_dis = 0;
    vector<int> order_nums;
    order_nums.push_back(0);

    for (int i = 1; i < v.size() - 1; i++) {
        for (int j = 0; j < orders.size(); j++) {
            if (orders[j].node_id == v[i]) {
                order_nums.push_back(orders[j].nums);
                break;
            }
        }
    }
    order_nums.push_back(0);

    for (int i = 1; i < v.size(); i++) {
        cur_load += order_nums[i];
        cur_dis += dis[v[i - 1]][v[i]];
        if (cur_dis + dis[v[i]][0] > 20 || cur_load > n) {
            sum += cur_dis - dis[v[i - 1]][v[i]] + dis[v[i - 1]][0];
            cur_load = order_nums[i];
            cur_dis = dis[0][v[i]];
        }
    }
    sum += cur_dis;

    return sum;
}

//比较器
bool cmp(const solution &a, const solution &b) {
    return a.sum < b.sum;
}

//初始化数据
void initData() {
    // 生成卸货点坐标, k = 64
    vector<node> unload_points;
    int id = 0;
    unload_points.push_back(node{0, 7, 7});
    for (int i = 0; i <= 14; i+=2) {
        for (int j = 0; j <= 14; j+=2) {
            id++;
            unload_points.push_back(node{id, i, j});
        }
    }
    
    static int order_id = 1; // 订单ID从1开始递增
    srand(time(0)); // 随机数种子
    for (const auto& point : unload_points) {
        int num_orders = rand() % (m + 1); // 每个卸货点随机生成0-5个订单
        if (num_orders != 0) {
            int priority = rand() % 3 + 1; // 随机生成1, 2, 或 3优先级
            int delay_total = 0, delay_cur = 0;

            if (priority == 1) delay_total = 0;
            else if (priority == 2) delay_total = 2;
            else if (priority == 3) delay_total = 5;

            for(int j = 0; j < orders_last.size(); j++) {
                if (orders_last[j].node_id == point.num) {
                    priority = min(priority, orders_last[j].priority);
                    num_orders += orders_last[j].nums;
                    delay_cur = max(delay_cur, orders_last[j].delay_cur);
                    delay_total = min(delay_total, orders_last[j].delay_total);
                    break;
                }
            }
            orders.push_back({order_id++, priority, num_orders, point.num, delay_cur, delay_total});
            city.push_back({point.num, point.x, point.y});
        }   
    }
    orders_last.clear();

    city.push_back({0, 7, 7});
    //计算距离, 存入unordered_map
    for (auto it = city.begin(); it != city.end(); it++) {
        unordered_map<int, double> temp;
        for (auto iterator = city.begin(); iterator != city.end(); iterator++) {
            temp[iterator->num] = distanceBetween(*it, *iterator);
        }
        dis[it->num] = temp;
    }

    for (auto it = city.begin(); it != city.end(); it++) {
        if (it->num != start) {
            num.push_back(it->num);
        }
    }
}

//初始化种群
void initPack(int gen) {
    for (int i = 0; i < packNum; i++) {
        //生成随机序列
        solution temp;
        temp.path.push_back(start);
        random_shuffle(num.begin(), num.end());
        for (auto it = num.begin(); it != num.end(); it++) {
            temp.path.push_back(*it);
        }
        temp.path.push_back(start);

        //计算个体的解并将其压入种群
        temp.gen = gen;
        temp.sum = sum(temp.path);
        pack.push_back(temp);
    }
    if (gen == 1) {
        solution greed;
        unordered_map<int, bool> isVisited;
        greed.path.push_back(start);
        isVisited[start] = true;
        for (int i = 0; i < num.size(); i++) {
            double min = -1;
            for (auto it = dis[greed.path.back()].begin(); it != dis[greed.path.back()].end(); it++) {
                if (isVisited.count(it->first) == 0 && (min == -1 || it->second < dis[greed.path.back()][min])) {
                    min = it->first;
                }
            }
            greed.path.push_back(min);
            isVisited[min] = true;
        }
        greed.path.push_back(start);
        greed.gen = gen;
        greed.sum = sum(greed.path);
        pack[0] = greed;
    }
}

//传代
void passOn() {
    record.push(pack[0].sum);
    //对每一代的最优解进行记录, 如果超过1000代相同则进行传代操作
    if (record.size() > 500) {
        record.pop();
        //传代后第一次变化前不进行传代，避免过早指数爆炸
        if (curCost != pack[0].sum && record.front() == record.back()) {
            //cout << "Pass On!" << endl;
            vector<int> Prometheus = pack[0].path;
            int gen = pack[0].gen;
            double sum = pack[0].sum;
            pack.clear();
            //在最大传代限制前进行种群扩增
            if (Hayflick > 0) {
                Hayflick--;
                packNum *= log(packNum) / log(10);
            }
            //重新初始化种群，并将火种压入
            initPack(gen);
            pack[0].path = Prometheus;
            pack[0].sum = sum;
            sort(pack.begin(), pack.end(), cmp);
            curCost = sum;
            mutationP += 0.1;
            //清空记录
            while (!record.empty()) {
                record.pop();
            }
        }
    }
}

//交叉算子
solution cross(solution &firstParent, solution &secondParent) {
    //随机选择长度与起点
    int length = int((rand() % 1000 / 1000.0) * city.size());
    int off = rand() % city.size() + 1;
    vector<int> nextGen(firstParent.path.size());
    unordered_map<int, bool> selected;
    nextGen[0] = start;
    for (int i = off; i < nextGen.size() - 1 && i < off + length; i++) {
        nextGen[i] = firstParent.path[i];
        selected[nextGen[i]] = true;
    }
    for (int i = 1, j = 1; i < nextGen.size(); i++) {
        if (nextGen[i] == 0) {
            for (; j < secondParent.path.size(); j++) {
                if (selected.count(secondParent.path[j]) == 0) {
                    nextGen[i] = secondParent.path[j];
                    selected[secondParent.path[j]] = true;
                    break;
                }
            }
        }
    }
    return solution{nextGen, sum(nextGen), 0, 0, firstParent.gen + 1};
}

//变异算子 - 顺序移位 - 优化速度
void mutation(solution &cur) {
    //随机选择长度与起点
    int length = int((rand() % 1000 / 1000.0) * city.size());
    int off = rand() % city.size() + 1;
    vector<int> m;
    unordered_map<int, bool> selected;
    m.push_back(start);
    for (int i = off; i < cur.path.size() - 1 && i < off + length; i++) {
        m.push_back(cur.path[i]);
        selected[cur.path[i]] = true;
    }
    for (int i = 1; i < cur.path.size(); i++) {
        if (selected.count(cur.path[i]) == 0) {
            m.push_back(cur.path[i]);
        }
    }
    for (int i = 0; i < m.size(); i++) {
        cur.path[i] = m[i];
    }
    cur.sum = sum(cur.path);
}

//变异算子 - 贪婪倒位 - 优化效率
void gmutation(solution &cur) {
    //随机选择起点，找到距起点最近的点，然后将最近点到起点之间的段倒位
    int selected = rand() % (city.size() - 4) + 2, min = 1;
    int selectedCity = cur.path[selected];
    int begin = 0, end = 0;
    for (auto it = dis[selectedCity].begin(); it != dis[selectedCity].end(); it++) {
        if (it->first != selectedCity && it->second < dis[selectedCity][min]) {
            min = it->first;
        }
    }
    for (int i = 1; i < cur.path.size() - 1; i++) {
        if (cur.path[i] == min) {
            if (i > selected + 1) {
                begin = selected + 1;
                end = i;
            } else if (i < selected - 1) {
                begin = i;
                end = selected - 1;
            }
            break;
        }
    }
    vector<int> stack;
    for (int i = begin; i <= end; i++) {
        stack.push_back(cur.path[i]);
    }
    for (int i = begin; i <= end; i++) {
        cur.path[i] = stack.back();
        stack.pop_back();
    }
    cur.sum = sum(cur.path);
}

//进化过程
vector<solution> process() {
    double total = 0;   //总适应度
    vector<solution> nextGenPack;   //下一代种群
    sort(pack.begin(), pack.end() - 1, cmp);    //排序找出最优个体
    //printf("%d %d\n", pack[0].gen, (int) pack[0].sum);

    passOn();   //传代操作
    //计算种群种每个个体的适应度
    for (auto it = pack.begin(); it != pack.end() - 1; it++) {
        it->F = 1 / it->sum;
        it->P = (it == pack.begin() ? 0 : (it - 1)->P) + it->F;
        total += it->F;
    }
    //最优个体直接进入下一代，防止反向进化
    nextGenPack.push_back(pack[0]);
    nextGenPack[0].gen++;
    //应用轮盘赌，选择交叉个体
    //由于种群容量不变且最优个体占位，故每一代保留最优个体的同时，剔除最差个体
    //即最差个体不参与轮盘赌交叉
    for (auto firstParent = pack.begin(); firstParent != pack.end() - 1; firstParent++) {
        if (rand() % 10000 / 10000.0 < crossP) {
            double selected = (rand() % 10000 / 10000.0) * total;
            for (auto secondParent = pack.begin(); secondParent != pack.end() - 1; secondParent++) {
                if (selected < secondParent->P) {
                    nextGenPack.push_back(cross(*firstParent, *secondParent));
                    break;
                }
            }
        } else {
            firstParent->gen++;
            nextGenPack.push_back(*firstParent);
        }
        if (rand() % 10000 / 10000.0 < mutationP) {
            //当传代1次后变更算子到更高效的贪婪倒位
            Hayflick < 6 ? gmutation(nextGenPack.back()) : mutation(nextGenPack.back());
        }
    }
    return nextGenPack;
}

void print(vector<int> &v) {
    double sum = 0;
    int cur_load = 0;
    double cur_dis = 0;
    int drone_id = 1;
    vector<int> order_nums;

    order_nums.push_back(0);
    printf("%d ", 0);
    for (int i = 1; i < v.size() - 1; i++) {
        for (int j = 0; j < orders.size(); j++) {
            if (orders[j].node_id == v[i]) {
                order_nums.push_back(orders[j].nums);
                printf("%d ", orders[j].nums);
                break;
            }
        }
    }
    printf("%d\n", 0);
    order_nums.push_back(0);

    printf("Drone%d\t: 0 ", drone_id);
    vector<Order> order_t;
    for (int i = 1; i < v.size(); i++) {
        cur_load += order_nums[i];
        cur_dis += dis[v[i - 1]][v[i]];
        
        if (cur_dis + dis[v[i]][0] > 20 || cur_load > n) {
            if (cur_load - order_nums[i] < 10 && cur_dis - dis[v[i - 1]][v[i]] + dis[v[i - 1]][0] < 15) {
                bool canDelay = true;
                for (int j = 0; j < order_t.size(); j++) {
                    if (order_t[j].delay_cur >= order_t[j].delay_total) {
                        canDelay = false;
                        break;
                    }         
                }
                if (canDelay) {
                    for (int j = 0; j < order_t.size(); j++) {
                        order_t[j].delay_cur++;
                        orders_last.push_back(order_t[j]);
                    }
                    printf("0, xx");
                } 
                order_t.clear();
            } else {
                printf("0,   ");
            }

            printf("\tdistance: %2f, load: %d\n", cur_dis - dis[v[i - 1]][v[i]] + dis[v[i - 1]][0], cur_load - order_nums[i]);
            sum += cur_dis - dis[v[i - 1]][v[i]] + dis[v[i - 1]][0];
            cur_load = order_nums[i];
            cur_dis = dis[0][v[i]];

            drone_id++;
            printf("Drone%d\t: 0 %d ", drone_id, v[i]);
            for (int j = 0; j < orders.size(); j++) {
                if (orders[j].node_id == v[i]) {
                    order_t.push_back(orders[j]);
                    break;
                }
            }
        } else {
            for (int j = 0; j < orders.size(); j++) {
                if (orders[j].node_id == v[i]) {
                    order_t.push_back(orders[j]);
                    break;
                }
            }
            printf("%d ", v[i]);
        }
    }
    sum += cur_dis;
    printf("\tdistance: %2f, load: %d\n", cur_dis, cur_load);

    if(orders_last.size() != 0) {
        printf("Delay Order: ");
        for(int j = 0; j < orders_last.size(); j++) {
            printf("%d(%d) ", orders_last[j].id, orders_last[j].node_id);
        }
        printf("\n");
    }
    
    return;
}

int main() {
    vector<double> clk;
    for(int i = 0; i < 100; i++) {
        clock_t start = clock();
        orders.clear();
        city.clear();
        num.clear();
        pack.clear();
        dis.clear();
        while(!record.empty()) record.pop();

        srand(unsigned(time(NULL)));    //设置时间种子
        initData();     //初始化数据
        initPack(1);    //初始化种群
        while (curGen <= maxGen) {
            pack = process();
            curGen++;
        }
        double temp = double(clock() - start) / CLOCKS_PER_SEC;
        clk.push_back(temp);
        cout << "Total time: " << temp << endl;

        cout << "The best: " << pack[0].sum << endl;
        for (auto it = pack[0].path.begin(); it != pack[0].path.end(); it++) {
            cout << *it << " ";
        }
        cout << endl;
        print(pack[0].path);
        
    }
    double avg = 0;
    for(int i = 0; i < clk.size(); i++) {
        avg += clk[i];
        printf("Time%d: %f\n", i, clk[i]);
    }
    printf("Average: %f", avg / 100);

    return 0;
}