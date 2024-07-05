clc
clear
close all;

data = xlsread('输入数据.xlsx',1,'A2:D31');
a1=data(:,2:3);

%% 遗传算法 初始化
cnum= 6;              %调整配送中心的数量  【可调参数】
dim = 30;            % 个体包含30位基因    【这里的基因长度=卸货点（卸货点）个数】
popsize = 10;        % 种群大小            【可调参数】
genmax = 150;        % 最大搜索次数        【可调参数】
Pm = 0.01;           % 变异概率
Pc = 0.85;              % 交叉概率

for i = 1 : popsize
    Parents(i,:) = [round((cnum-1)*rand(1,dim-cnum))+1 , randperm(cnum,cnum) ]; %随机生成初始种群，popsize个个体
end

%% 计算初代适应度值
bb0 = data;
[fitness,distance] =ga_fitness(bb0,popsize,Parents,a1,cnum); %调用适应度值函数
%% 每一代的个体平均适应度值
every_generation_fitness = zeros(1, genmax+1);
repeat_num_all = zeros(1, genmax);               % 每一代种群中重复个体的数量
current_generation_fitness = fitness;
every_generation_fitness(1,1) = sum(fitness(1,:))/popsize;% 保存每一代的个体平均适应度值

%% 搜索开始
for t = 1 : genmax
    [population]=ga_cross(Parents,Pc);      % 交叉
    [Children]=ga_mut(population, Pm,cnum); % 变异
    Combination = [Parents;Children];        % 合并子代和父代
    [Parents,fit_parent]=ga_TourSel(bb0,popsize,Combination,a1,cnum); % 锦标赛选择,适应度计算在优选内完成
    every_generation_fitness(1,t+1) = sum(fit_parent(1,:))/popsize; % 父代种群适应度和÷种群大小=个体平均适应度值
    data1=sortrows(Parents);  % 种群中相同的个体
    [~,r]=unique(data1,'rows');
    [dataUnique,r1]=unique(data1,'rows','last');
    repeat_num_all(1,t) = max(r1-r+1);
end

%% 绘图
figure();
x =0:1:genmax;
plot( x , every_generation_fitness,'k-o','linewidth',1);   % x为向量时，则以x为横坐标，y中元素为纵坐标显示，若x与y为同维矩阵，则将x和y对应位置上的元素作为横纵坐标绘制图像
xlabel('搜索次数');       % 标识横坐标
ylabel('个体平均适应度值');       % 标识横坐标

% 绘图
figure();
x2 =1:1:genmax;
plot( x2 , repeat_num_all,'r-*','linewidth',1);   % x为向量时，则以x为横坐标，y中元素为纵坐标显示，若x与y为同维矩阵，则将x和y对应位置上的元素作为横纵坐标绘制图像
xlabel('搜索次数');               % 标识横坐标
ylabel('种群中相同个体的个数');    % 标识横坐标

%% 得到最优个体
[fit_final,distance_02,V0_fit, ind1,C2C] =ga_fitness(bb0,popsize,Parents,a1,cnum); %调用适应度值函数

[min_p,pos] = min(fit_final);
V0=V0_fit; %聚类中心的坐标
best_ind=ind1;
%计算中心坐标更新后的距离
bb01=[bb0(:,1:3),best_ind];
for ii=1 : dim %dim
    for j=1:size(V0,1) % cnum
        if bb01(ii,4)==j
            distance_fit(ii,:)= calculateDistance(bb01(ii,2), bb01(ii,3), V0(j,1), V0(j,2));
        else
            continue
        end
    end
end
Trans_time_fit = distance_fit./60*60; %计算运输时间  运输速度60km/h 再转化为分钟min

%% 结果输出
figure()
index = 1:1:30;
h = result_plot(index,a1,best_ind,V0,cnum);
legend([h(1),h(2),h(3),h(4),h(5),h(6)], {'1号配送中心','2号配送中心','3号配送中心','4号配送中心','5号配送中心','6号配送中心'});
xlim([0 22])
hold off

%% final_data
bb2=[data,ind1,distance_fit,Trans_time_fit];
%1        2      3        4        5            6             7
%序号   x坐标  y坐标   需求量  配送中心编号   拒配送中心距离  救援时间

%% 判断配送中心服务的卸货点数量是否超出限制
num_limit = 15; % 一个配送中心  所能服务的卸货点数量上限
num_judge = bb2(:,[1,5]);
tmp=sortrows(num_judge,2);
[p1,p2]= unique(tmp(:, 2));
num_tmp = zeros(1,length(p1));
for i = 1 : length(p1)
    if i ~= length(p1)
        num_tmp(i)=p2(i+1)-p2(i);
    else
        num_tmp(i)=length(num_judge)+1-p2(i);
    end
    if num_tmp(i) > 15
        disp([num2str(i),'号配送中心的负荷不满足要求:',num2str(num_tmp(i))])
    else
        disp([num2str(i),'号配送中心的负荷满足要求'])
    end
end
disp('-------------------------------------')
%% 运输里程总和
ee1=[bb2(:,5),bb2(:,4),distance_fit];%bb2的第五列是所属配送中心索引  bb2的第四列是需求量
firstCol = ee1(:, 1);
secondCol = ee1(:, 2);
thirdCol = ee1(:, 3);
% 使用accumarray进行分组求和
[uniqueKeys, ~, sub2ind] = unique(firstCol);
sums1 = accumarray(sub2ind, secondCol, [], @sum);
sums2 = accumarray(sub2ind, thirdCol, [], @sum);
% 结果是一个向量，其中索引i处的值是第一列中值为uniqueKeys(i)的所有对应行的第二列的和
for i=1:size(sums2,1)
    disp([num2str(i),'号配送中心距离所有卸货点总里程：',num2str(sums2(i))])
end
disp('-------------------------------------')

%% 输出配送中心选址结果
disp('卸货点距离所属的配送中心距离为：')
disp(bb2(:,6))
disp(['一共有',num2str(length(V0)),'个配送中心,选址坐标为：'])
disp(V0)
disp('-------------------------------------')
disp(['距离配送中心总距离为：',num2str(sum(bb2(:,6))),'km'])
disp(['直接从配送中心出发总配送时间为：',num2str(sum(bb2(:,7))),'min'])
disp('-------------------------------------')

for i = 1:cnum    %输出每一类的样本点标号
    str=['第' num2str(i) '个配送中心辖区内的卸货点有:  ' num2str(C2C{i})];
    disp(str);
end
disp('-------------------------------------')

%% 生成随机订单
%随机生成订单 （此处假设一个卸货点每次仅生成一个订单）
% 	1 紧急：  0.5小时内配送到
% 	2 较紧急：1.5小时内配送到
% 	3 一般：  3.0小时内配送到
demonds  = [round((3-1)*rand(1,dim-3))+1 , randperm(3,3)]'; %每次执行随机生成的订单紧急程度
t_limit = zeros(dim,1);%设置时间窗约束
for i = 1:dim
    if  demonds(i)==1
        t_limit(i)=30;
    elseif  demonds(i)==2
        t_limit(i)=90;
    elseif demonds(i)==3
        t_limit(i)=180;
    end
end

for i = 1:cnum
    % 确定当前类别的数据点
    data_tem = [ind1, a1, bb0(:,1),demonds,bb2(:,6)];
    data_tem = sortrows(data_tem, 1);
    % 找出分类边界
    [x1, x2] = unique(data_tem(:, 1));
    if i < length(x1)
        y1_tem = data_tem(x2(i):x2(i+1)-1,:);
    else
        y1_tem = data_tem(x2(i):end,:);
    end
    y1_tem = sortrows(y1_tem, [5,6]); %先按照订单优先级排序，再按照距离来排序
    C_all{i} = y1_tem;
end

%% 分配运输路径
distance = cell(cnum, 1);
for i =1:cnum
    aa = i;
    c_tmp = C_all{i};
    index = c_tmp(:,4);
    figure()
    %legendData = {[num2str(i)','号配送中心']};
    result_plot(index,c_tmp(:,2:3),c_tmp(:,1),V0(i,:),cnum);
    %根据距离判断无人机数量
    if 2*sums2(i) > 20
        vc_num = ceil(2*sums2(i)/20);
    else
        vc_num = 1;
    end
    title([num2str(i),'号配送中心运输路线图，无人机数量：',num2str(vc_num)])
    % 起点的坐标
    startPoint =V0(i,:);
    if vc_num == 1
        VC = {c_tmp(:,4)'};
        mat1 =V0(i,:);
        mat2 =c_tmp(:,2:3);
        result = [mat1; mat2];
        vertexs = {result};
        draw_vcRoute(VC,vertexs,aa)
        % 点的坐标矩阵
        points = mat2;
        % 计算从起点到每个点的距离
        dist0= sqrt(sum((points(:, 1) - startPoint(1)).^2 + (points(:, 2) - startPoint(2)).^2, 2));
        % 根据距离和速度计算行走时间
        dis_tmp1 = {dist0};
    else
        VC = cell(vc_num, 1);
        dis_tmp0 = cell(vc_num, 1);
        for jj=1:vc_num
            c_tmpi= c_tmp(jj:vc_num:end,:);
            VC(jj) = {c_tmpi(:,4)'};
            vertexs{jj} = [V0(i,:);c_tmpi(:,2:3)];
            points = c_tmpi(:,2:3);
            dis_tmp = sqrt(sum((points(:, 1) - startPoint(1)).^2 + (points(:, 2) - startPoint(2)).^2, 2));
            dis_tmp0(jj)= {dis_tmp};
        end
        dis_tmp1 = {dis_tmp0};
        draw_vcRoute(VC,vertexs,aa)
    end
    VC_all{i} = VC;
    clear VC
    clear vertexs
    hold off
    distance(i)=dis_tmp1;
end

%% 判断配送中心是否满足所有卸货点的时间窗约束
%取出每个点的运输距离
d_final = [];
for i = 1:cnum
    tmp =  distance{i};
    if iscell(tmp)
        tmp0 =  tmp;
        tmp = [];
        for j = 1:length(tmp0)
            tmp1 =  tmp0{j};
            tmp =[tmp;tmp1];
        end
    end
    d_final = [d_final;tmp];
end
t_final = d_final./60*60;

err_num=[];
for i = 1 : dim
    if t_final(i) < t_limit(i)
        err_num(i)=0;
        continue
    else
        err_num(i)=1;
        break
    end
end
if sum(err_num)==0
    disp('所有配送中心均能够满足辖区内卸货点的时间窗约束')
else
    disp([num2str(i),'号配送中心不满足时间窗要求'])
end