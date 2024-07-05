function [fitness, distance, V0_fit, ind1,C] = ga_fitness(bb0, popsizeeee, Parents, a1, cnum)
%[fitness,distance] =ga_fitness(bb0,popsize,Parents,a1,cnum); %调用适应度值函数
% popsizeeee=popsize;

% 初始化输出变量
fitness = zeros(1, popsizeeee);
distance = zeros(size(bb0, 1), popsizeeee);
for jj = 1:popsizeeee
    % 初始化V0
    V0 = zeros(cnum, 2);
    % 获取当前父母的染色体
    Parents_01 = Parents(jj,:)';
    data_tem = [Parents_01, a1, bb0(:,1)];
    data_tem = sortrows(data_tem, 1);
    % 找出分类边界
    [x1, x2] = unique(data_tem(:, 1));
    % 确保有足够的起降点数量
    if length(x1) < cnum
        Parents(jj,:)=[round((cnum-1)*rand(1,30-cnum))+1 , randperm(cnum,cnum)];
        Parents_01 = Parents(jj,:)';
        data_tem = [Parents_01, a1, bb0(:,1)];
        data_tem = sortrows(data_tem, 1);
        % 找出分类边界
        [x1, x2] = unique(data_tem(:, 1));
    end
    % 计算聚类中心
    for i = 1:cnum
        % 确定当前类别的数据点
        if i < length(x1)
            y1_tem = data_tem(x2(i):x2(i+1)-1, 2:3);
        else
            y1_tem = data_tem(x2(i):end, 2:3);
        end
        % 使用fminsearch找到初始聚类中心
        points = y1_tem;
        objectiveFunction = @(x) sum(sqrt(sum((points - x(ones(size(points,1),1),:)).^2, 2)));
        initialGuess = [mean(points(:,1)), mean(points(:,2))];
        [V0(i,:), dis1] = fminsearch(objectiveFunction, initialGuess);
    end
    % 计算每个点到聚类中心的距离
     bb01=[bb0(:,1:3),Parents_01];
    for ii = 1:size(bb0, 1)
        for j = 1:cnum
            if bb01(ii, 4) == j
                distance(ii, jj) = calculateDistance(bb01(ii, 2), bb01(ii, 3), V0(j, 1), V0(j, 2));
            end
        end
    end
    % 计算fitness（即所有距离的总和）
    fitness(jj) = sum(distance(:, jj));
end
    
    [B,C,CC]  =   k_means(2,cnum,a1);
    V0_fit = CC; 
    ind1= B';
end

