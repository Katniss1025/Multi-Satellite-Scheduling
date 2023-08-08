% 定义球的半径
radius = 6371393;

% 生成球的网格
[X, Y, Z] = sphere(100);

% 缩放球的尺寸
X = X * radius;
Y = Y * radius;
Z = Z * radius;

% 画球体
figure;
surf(X, Y, Z);
hold on,

% 画中点
x = 4251472.13012989;
y = -5398242.9073892;
z = 0;
scatter3(x,y,z,'filled');
hold on,

% 画目标
x1 = 5e7 * -0.61540273;
y1 = 5e7 * 0.78139838;
z1 = 5e7 * -0.1034217;
scatter3(x,y,z,'filled','g');
hold on,

% 画轨道
R = 6871393.0; 
theta = linspace(0, 2*pi, 100);  % 角度范围从 0 到 2pi，将圆分为 100 个点
x = R * cos(theta);
y = R * sin(theta);

% 画连线
plot(x, y, 'r', 'LineWidth', 2);
hold on,

% 画进出点
% angle1 = 89.99936208147079;
% enter_x = R * cosd(angle1);
% enter_y = R * sind(angle1);
% angle2 = 266.89309098095237;
% exit_x = R * cosd(angle2);
% exit_y = R * sind(angle2);
% scatter3(enter_x,enter_y,0,'filled','black');
% hold on,
% scatter3(exit_x,exit_y,0,'filled','black');
% hold on,

for i = 1:1:100
    plot3([x1, x(i)], [y1, y(i)], [z1, 0], 'r', 'LineWidth', 2);
    hold on,
end

axis equal;

