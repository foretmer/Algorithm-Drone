%% 画出最优配送方案路线图
%输入：VC              配送方案
%输入：vertexs0         各个点的坐标
function draw_vcRoute(VC,vertexs,aa)
disp([num2str(aa),'号配送中心运输方案：'])
NV=size(VC,1);                                                  %无人机使用数目
% hold on;box on
xlabel('x坐标')
ylabel('y坐标')
hold on;
C=hsv(NV);
for i=1:NV
    part_seq=VC{i};            %每辆车所经过的卸货点
    tmp = vertexs{i};
    customer = tmp(2:end,:);
    vertexs0 = tmp(1,:);
    len=length(part_seq);                           %每辆车所经过的卸货点数量
    for j=0:len
        %当j=0时，无人机从配送中心出发到达该路径上的第一个卸货点
        if j==0
            fprintf('%s','   配送路线',num2str(i),'：');
            fprintf('%d->',0);
            
            c1=customer(1,:);
            plot([vertexs0(1,1),c1(1)],[vertexs0(1,2),c1(2)],'-','color',C(i,:),'linewidth',1);
            quiver(vertexs0(1,1), vertexs0(1,2), [c1(1) - vertexs0(1,1)], [c1(2) - vertexs0(1,2)],...
                'Color', C(i,:), 'LineStyle', '-.','linewidth',1)
            %当j=len时，无人机从该路径上的最后一个卸货点出发到达配送中心
        elseif j==len
            fprintf('%d->',part_seq(j));
            fprintf('%d',0);
            fprintf('\n');
            c_len=customer(end,:);
            plot([c_len(1),vertexs0(1,1)],[c_len(2),vertexs0(1,2)],'-','color',C(i,:),'linewidth',1);
            quiver(c_len(1), c_len(2), [vertexs0(1,1) - c_len(1)], [vertexs0(1,2) - c_len(2)],...
                'Color', C(i,:), 'LineStyle', '-.','linewidth',1)
            %否则，无人机从路径上的前一个卸货点到达该路径上紧邻的下一个卸货点
        else
            fprintf('%d->',part_seq(j));
            c_pre=customer(j,:);
            c_lastone=customer(j+1,:);
            plot([c_pre(1),c_lastone(1)],[c_pre(2),c_lastone(2)],'-','color',C(i,:),'linewidth',1);
             quiver(c_pre(1), c_pre(2), [c_lastone(1) - c_pre(1)], [c_lastone(2) - c_pre(2)],...
             'Color', C(i,:), 'LineStyle', '-','linewidth',1);
        end
    end
end
% plot(customer(:,1),customer(:,2),'ro','linewidth',1);hold on;
% plot(vertexs0(1,1),vertexs0(1,2),'s','linewidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',10);
end

