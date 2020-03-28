
% Konvoluvcija: preprosti primeri z for zanko
%
% originalna formula za konvolucijo (samo vzro�ni del):
%
%             inf
%            ---
%     y(n) = \    x(k)*h(n-k)            za n = 0,1,2,...
%            /
%            ---
%             k=0
%
% v na�em primeru je prvi element x oz. h na indeksu 1 (in ne na 0, kot je to v zgornji formuli), torej
%
%             inf
%            ---
%     y(n) = \    x(k+1)*h(n-k)          za n = 1,2,3,...
%            /
%            ---
%             k=0
%
% OPOMBA: pri n-k se vpliv postavitve za�etnega indeksa izni�i: n+1 - (k+1) = n-k
%
% n je smiselno omejiti z zgornjo mejo length(x)+length(h)-1, saj so naprej same ni�le...
% zaradi h(n-k) mora biti n-k med 1 in length(h), torej mora biti k med n-length(h) in n-1, ampak samo za pozitivne n-k!
% zaradi x(k+1) mora teci k med 0 in length(x)-1

clc;
x = [1 2 3 4 3 2 1];
h = [1 2 1]; 

y = zeros(1,length(x)+length(h)-1); % dolzina izhoda
for n = 1:length(x)+length(h)-1
    disp('...............')
    disp(['  n = ' num2str(n)]);
    for k =  max( [n-length(h),0] ) : min([n-1,length(x)-1]) % bodimo pozorni na meje, ki upostevajo, da se prvi element vektroja zacne pri indeksu 1 !!!
        disp(['     k = ' num2str(k)]);
        disp(['        n-k = ' num2str(n-k)]);
        y(n) = y(n) + x(k+1)*h(n-k);        
    end
end

y2 = conv(x,h);

figure; 
plot(y,'LineWidth',2); 
hold on; 
plot(y2,'r:','LineWidth',2);
xlabel('vzorci');
ylabel('amplituda')
axis tight;
legend('for zanka','conv')


% Konvoluvcija: preprosti primeri s for zanko 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
close all;
x = [ zeros(1,50) 1 zeros(1,50)];
h = [0:0.1:1 1:-0.025:0];

figure;
subplot(2,1,1);  plot(x,'b','LineWidth',2);
axis tight; xlabel('vzorci'); ylabel('amplituda'); title('vhod (x)');
subplot(2,1,2);  plot(h,'g','LineWidth',2);
axis tight; xlabel('vzorci'); ylabel('amplituda'); title('odziv (h)');

figure; hold on
y = zeros(1,length(x)+length(h)-1); % dolzina izhoda
for n = 1:length(x)+length(h)-1
    for k =  max( [n-length(h),0] ) : min([n-1,length(x)-1]) % bodimo pozorni na meje, ki upostevajo, da se prvi element vektroja zacne pri indeksu 1 !!!
        y(n) = y(n) + x(k+1)*h(n-k);
    end
    cla;
    plot(x,'b','LineWidth',2);
    plot(n-[length(h)-1:-1:0],h(end:-1:1),'g','LineWidth',2);
    plot(y,'r','LineWidth',2);
    axis tight; xlabel('vzorci'); ylabel('amplituda');
    title('vhod (x) - modro, odziv (h) - zeleno, izhod (y) - rde�e')
    pause(0.20);
end
close;

%%%%%%%%%%%%%%%%%%%%%%
figure; 
subplot(3,1,1); plot(x,'LineWidth',2);
axis tight; title('x');
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,2); plot(h,'g','LineWidth',2);
axis tight; title('h')
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,3); plot(y,'r','LineWidth',2);
axis tight; title('y')
xlabel('vzorci');
ylabel('amplituda');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = 30; % razmik impulzov v x
x = zeros(1,100);
x(1:d:100) = 1;
h = [0:0.1:1 1:-0.025:0];
x(1)=2

figure; 
subplot(2,1,1);  plot(x,'b','LineWidth',2);
axis tight; xlabel('vzorci'); ylabel('amplituda'); title('vhod (x)'); 
subplot(2,1,2);  plot(h,'g','LineWidth',2);
axis tight; xlabel('vzorci'); ylabel('amplituda'); title('odziv (h)'); 

figure; hold on
y = zeros(1,length(x)+length(h)-1); % dolzina izhoda
for n = 1:length(x)+length(h)-1
    for k =  max( [n-length(h),0] ) : min([n-1,length(x)-1]) % bodimo pozorni na meje, ki upostevajo, da se prvi element vektroja zacne pri indeksu 1 !!!
        y(n) = y(n) + x(k+1)*h(n-k);
    end
    cla;
    plot(x,'b','LineWidth',2);
    plot(n-[length(h)-1:-1:0],h(end:-1:1),'g','LineWidth',2);
    plot(y,'r','LineWidth',2);
    axis tight; xlabel('vzorci'); ylabel('amplituda');
    title('vhod (x) - modro, odziv (h) - zeleno, izhod (y) - rde�e')
    pause(0.20);
end
close;

%% %%%%%%%%%
figure; 
subplot(3,1,1); plot(x,'LineWidth',2);
axis tight; title('x');
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,2); plot(h,'g','LineWidth',2);
axis tight; title('h')
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,3); plot(y,'r','LineWidth',2);
axis tight; title('y')
xlabel('vzorci');
ylabel('amplituda');


% Konvoluvcija: preprosti primeri s funkcijo conv()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [ zeros(1,50) 1 zeros(1,50)];
h = [0:0.1:1 1:-0.025:0];
%h = ones(1,22);

figure; 
subplot(3,1,1); plot(x,'LineWidth',2);
axis tight; title('x');
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,2); plot(h,'g','LineWidth',2);
axis tight; title('h')
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,3); plot(conv(x,h),'r','LineWidth',2);
axis tight; title('conv(x,h)')
xlabel('vzorci');
ylabel('amplituda');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [ zeros(1,50) 1 zeros(1,25) 1  zeros(1,50)];
h = [0:0.1:1 1:-0.025:0];
%h = ones(1,22);

figure; 
subplot(3,1,1); plot(x,'LineWidth',2);
axis tight; title('x');
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,2); plot(h,'g','LineWidth',2);
axis tight; title('h')
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,3); plot(conv(x,h),'r','LineWidth',2);
axis tight; title('conv(x,h)')
xlabel('vzorci');
ylabel('amplituda');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bolj kompleksen primer....
d = 20; % razmik impulzov v x
x = [ zeros(1,50) 1 zeros(1,d) 1 zeros(1,d) 1  zeros(1,50)];
%h = [0:0.1:1 1:-0.025:0];
h = rand(1,30);

figure; 
subplot(3,1,1); plot(x,'LineWidth',2);
axis tight; title('x');
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,2); plot(h,'g','LineWidth',2);
axis tight; title('h')
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,3); plot(conv(x,h),'r','LineWidth',2);
axis tight; title('conv(x,h)')
xlabel('vzorci');
ylabel('amplituda');


% ALGEBRAI�NE LASTNOSTI KONVOLUCIJE
% KOMUTATIVNOST
%     f * g = g * f ,
% 
% ASOCIATIVNOST
%     f * (g * h) = (f * g) * h ,
% 
% DISTRIBUTIVNOST
%     f * (g + h) = (f * g) + (f * h) ,
% 
% ASOCIATIVNOST S SKALARNIM MNO�ENJEM
%     a (f * g) = (a f) * g = f * (a g) ,
% 
% KOMUTATIVNOST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     x * h = h * x ,
x = [ zeros(1,50) 1 zeros(1,50)];
h = [0:0.1:1 1:-0.025:0];
figure; 
subplot(2,1,1); plot(conv(x,h),'r','LineWidth',2);
axis tight; title('KOMUTATIVNOST KONVOLUCIJE: conv(x,h)');
xlabel('vzorci');
ylabel('amplituda');
subplot(2,1,2); plot(conv(h,x),'r','LineWidth',2);
axis tight; title('KOMUTATIVNOST KONVOLUCIJE: conv(h,x)')
xlabel('vzorci');
ylabel('amplituda');

% ASOCIATIVNOST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     g * (x * h) = (g * x) * h ,
x = [ zeros(1,50) 1 zeros(1,50)];
h = [0:0.1:1 1:-0.025:0];
g = sin(0:0.1:pi);
figure; 
subplot(2,1,1); plot(conv(g,conv(x,h)),'r','LineWidth',2);
axis tight; title('ASOCIATIVNOST KONVOLUCIJE: conv(g,conv(x,h)) ');
xlabel('vzorci');
ylabel('amplituda');
subplot(2,1,2); plot(conv(conv(g,x),h),'r','LineWidth',2);
axis tight; title('ASOCIATIVNOST KONVOLUCIJE: conv(conv(g,x),h) ')
xlabel('vzorci');
ylabel('amplituda');

% DISTRIBUTIVNOST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     x * (g + h) = (x * g) + (x * h) ,
x = [ zeros(1,50) 1 zeros(1,50)];
h = cos(0:0.05:pi);
g = sin(0:0.05:pi);
figure; 
subplot(2,1,1); plot(conv(x,(g+h)),'r','LineWidth',2);
axis tight; title('DISTRIBUTIVNOST KONVOLUCIJE:  conv(x,(g+h)) ');
xlabel('vzorci');
ylabel('amplituda');
subplot(2,1,2); plot(conv(x,g) + conv(x,h),'r','LineWidth',2);
axis tight; title('DISTRIBUTIVNOST KONVOLUCIJE: conv(x,g) + conv(x,h)  ')
xlabel('vzorci');
ylabel('amplituda');

% ASOCIATIVNOST S SKALARNIM MNO�ENJEM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     a (x * h) = (a x) * h = x * (a h) ,
x = [ zeros(1,50) 1 zeros(1,50)];
h = sin(0:0.05:pi);
a = randn;
figure; 
subplot(3,1,1); plot(a*conv(x,h),'r','LineWidth',2);
axis tight; title('ASOCIATIVNOST S SKALARNIM MNO�ENJEM:  a*conv(x,h) ');
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,2); plot(conv(a*x,h),'r','LineWidth',2);
axis tight; title('ASOCIATIVNOST S SKALARNIM MNO�ENJEM: conv(a*x,h)  ');
xlabel('vzorci');
ylabel('amplituda');
subplot(3,1,3); plot(conv(x,a*h),'r','LineWidth',2);
axis tight; title('ASOCIATIVNOST S SKALARNIM MNO�ENJEM: conv(x,a*h) ');
xlabel('vzorci');
ylabel('amplituda');


% Konvoluvcija in govor
% impulzni odzivi posameznih prostorov: http://www.voxengo.com/impulses/ 
%%%%%% posnamemo govor %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;
Fs = 44100;
bits = 16;
my_recorder = audiorecorder(Fs,bits,2);
recordblocking(my_recorder,1);
posnetek = getaudiodata(my_recorder);
sound(posnetek,Fs);

figure; 
subplot(2,1,1); plot([1:length(posnetek)]/Fs,posnetek(:,1));
axis tight; title('kanal 1')
xlabel('�as (s)');
ylabel('amplituda');
subplot(2,1,2); plot([1:length(posnetek)]/Fs,posnetek(:,2));
axis tight; title('kanal 2')
xlabel('�as (s)');
ylabel('amplituda');

%%%%%%%%%%%%% nalo�imo impulzni odziv sobe (dostopen na spletu) %%%%%%%%%%%%%%%%%%%%%%%%
% impulzni odzivi posameznih prostorov: http://www.voxengo.com/impulses/ 

[h, Fs] = audioread('.\IMreverbs\Going Home.wav');
%h = h / norm(h);   % ?e uporabljamo starej�o verzijo matlaba kjer nam vrne ne normaliziran vektor
sound(40*h,Fs); 

figure; 
subplot(2,1,1); plot([1:length(h)]/Fs,h(:,1),'k');
axis tight; title('kanal 1')
xlabel('�as (s)');
ylabel('amplituda');
subplot(2,1,2); plot([1:length(h)]/Fs,h(:,2),'k');
axis tight; title('kanal 2')
xlabel('�as (s)');
ylabel('amplituda');

%%%%%%%%%%%%% konvolucija v �asovni domeni s for zanko %%%%%%%%%%%%%%%%%%%%%%%%
clear efekt;

efekt(:,1) = zeros(1,length(posnetek)+length(h)-1);
efekt(:,2) = zeros(1,length(posnetek)+length(h)-1);

posnetekNorm = posnetek * norm(posnetek);

tic;
for n = 1:length(posnetek)+length(h)-1
    for k = max([n-length(h),0]) : min([n-1,length(posnetek)-1])
        efekt(n,1) = efekt(n,1) + posnetekNorm(k+1,1) * h(n-k,1);
        efekt(n,2) = efekt(n,2) + posnetekNorm(k+1,2) * h(n-k,2);
    end
end
toc;

sound(efekt,Fs);

figure; 
subplot(2,1,1); plot([1:length(efekt)]/Fs,efekt(:,1),'r'); 
hold on; plot([1:length(posnetek)]/Fs,posnetek(:,1));
axis tight; title('kanal 1')
xlabel('�as (s)');
ylabel('amplituda');
subplot(2,1,2);  plot([1:length(efekt)]/Fs,efekt(:,2),'r'); 
hold on; plot([1:length(posnetek)]/Fs,posnetek(:,2));
axis tight; title('kanal 2')
xlabel('�as (s)');
ylabel('amplituda');

%%%%%%%%%%%%% konvolucija v �asovni domeni z vektorsko operacijo %%%%%%%%%%%%%%%%%%%%%%%%
clear efekt;

tic;
efekt(:,1) = conv(posnetek(:,1),h(:,1));
efekt(:,2) = conv(posnetek(:,2),h(:,2));
toc;

sound(efekt,Fs);

figure; 
subplot(2,1,1); plot([1:length(efekt)]/Fs,efekt(:,1),'r'); 
hold on; plot([1:length(posnetek)]/Fs,posnetek(:,1));
axis tight; title('kanal 1')
xlabel('�as (s)');
ylabel('amplituda');
subplot(2,1,2);  plot([1:length(efekt)]/Fs,efekt(:,2),'r'); 
hold on; plot([1:length(posnetek)]/Fs,posnetek(:,2));
axis tight; title('kanal 2')
xlabel('�as (s)');
ylabel('amplituda');

%%%%%%%%%%%%%%% konvolucija v frekven�ni domeni z vektorsko operacijo %%%%%%%%%%%%%%%%%%%%%%%
clear efekt;

tic;
X = fft([posnetek(:,1); zeros(length(h(:,1))-1,1)]);
Y = fft([h(:,1); zeros(length(posnetek(:,1))-1,1)]);
efekt(:,1) = ifft(X.*Y);
toc;

tic
X = fft([posnetek(:,2); zeros(length(h(:,2))-1,1)]);
Y = fft([h(:,2); zeros(length(posnetek(:,2))-1,1)]);
efekt(:,2) = ifft(X.*Y);
toc

sound(efekt,Fs);

figure; 
subplot(2,1,1); plot([1:length(efekt)]/Fs,efekt(:,1),'r'); 
hold on; plot([1:length(posnetek)]/Fs,posnetek(:,1));
axis tight; title('kanal 1')
xlabel('�as (s)');
ylabel('amplituda');
subplot(2,1,2);  plot([1:length(efekt)]/Fs,efekt(:,2),'r'); 
hold on; plot([1:length(posnetek)]/Fs,posnetek(:,2));
axis tight; title('kanal 2')
xlabel('�as (s)');
ylabel('amplituda');



%%% kako pa je s konvolucijo v �asovnem prostoru, �e so signali dolgi 
[h, Fs] = audioread('.\IMreverbs\Five columns.wav');
%h = h / norm(h);   % ?e uporabljamo starej�o verzijo matlaba kjer nam vrne ne normaliziran vektor

%%%%%% posnamemo govor %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bits = 16;
my_recorder = audiorecorder(Fs,bits,2);
recordblocking(my_recorder,30);
posnetek = getaudiodata(my_recorder);
sound(posnetek,Fs);

%%%%%%%%%%%%%%% konvolucija v �asovni domeni %%%%%%%%%%%%%%%%%%%%%%%
clear efekt;

tic
efekt(:,1) = conv(posnetek(:,1),h(:,1));
efekt(:,2) = conv(posnetek(:,2),h(:,2));
toc

sound(efekt,Fs);

%%%%%%%%%%%%%%% konvolucija v frekven�ni domeni %%%%%%%%%%%%%%%%%%%%%%%
clear efekt;

tic;
X = fft([posnetek(:,1); zeros(length(h(:,1))-1,1)]);
Y = fft([h(:,1); zeros(length(posnetek(:,1))-1,1)]);
efekt(:,1) = ifft(X.*Y);
toc;

tic
X = fft([posnetek(:,2); zeros(length(h(:,2))-1,1)]);
Y = fft([h(:,2); zeros(length(posnetek(:,2))-1,1)]);
efekt(:,2) = ifft(X.*Y);
toc

sound(efekt,Fs);



% Konvoluvcija in 3D zvok
% impulzni odzivi posameznih pozicij v prostoru: http://recherche.ircam.fr/equipes/salles/listen/download.html 
%%%%%% posnamemo govor %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;
Fs = 44100;
bits = 16;
my_recorder = audiorecorder(Fs,bits,2);
recordblocking(my_recorder,5);
posnetek = getaudiodata(my_recorder);
sound(posnetek,Fs);

figure; 
subplot(2,1,1); plot([1:length(posnetek)]/Fs,posnetek(:,1));
axis tight; title('kanal 1')
xlabel('�as (s)');
ylabel('amplituda');
subplot(2,1,2); plot([1:length(posnetek)]/Fs,posnetek(:,2));
axis tight; title('kanal 2')
xlabel('�as (s)');
ylabel('amplituda');

%%%%%%%%%%%%% nalo�imo impulzni odziv 3D zvoka (dostopen na spletu) %%%%%%%%%%%%%%%%%%%%%%%%
% impulzni odzivi posameznih prostorov: http://recherche.ircam.fr/equipes/salles/listen/download.html 

load('.\3Dsound\IRC_1059_C_HRIR.mat')

elevation = 50; % elevation (in degrees) [-45,90];
azimuth = 270; % azimuth (in degrees) [0 360];
[pos_err,pos_ind] = min( abs(l_eq_hrir_S.elev_v - elevation) + abs(l_eq_hrir_S.azim_v - azimuth))

left_channel_IR  = l_eq_hrir_S.content_m(pos_ind,:);
right_channel_IR = r_eq_hrir_S.content_m(pos_ind,:); 

%sound(40*left_channel_IR,Fs); 

figure; 
subplot(2,1,1); plot([1:length(left_channel_IR)]/Fs*1000,left_channel_IR,'k');
axis tight; title('levi kanal');
xlabel('�as (ms)');
ylabel('amplituda');
subplot(2,1,2); plot([1:length(right_channel_IR)]/Fs*1000,right_channel_IR,'k');
axis tight; title('desni kanal');
xlabel('�as (ms)');
ylabel('amplituda');

%%%%%%%%%%%%%%% konvolucija v frekven�ni domeni %%%%%%%%%%%%%%%%%%%%%%%
clear posnetek3D;

% convert stereo to mono sound
posnetekMONO = (posnetek(:,1)+posnetek(:,2))/2;

% left channel
X = fft([posnetekMONO; zeros(length(left_channel_IR)-1,1)]);
Y = fft([left_channel_IR'; zeros(length(posnetekMONO)-1,1)]);
posnetek3D(:,1) = ifft(X.*Y);

% left channel
X = fft([posnetekMONO; zeros(length(right_channel_IR)-1,1)]);
Y = fft([right_channel_IR'; zeros(length(posnetekMONO)-1,1)]);
posnetek3D(:,2) = ifft(X.*Y);

figure; 
subplot(2,1,1); plot([1:length(posnetek3D)]/Fs,posnetek3D(:,1),'r'); 
hold on; plot([1:length(posnetek)]/Fs,posnetek(:,1));
axis tight; title('kanal 1')
xlabel('�as (s)');
ylabel('amplituda');
subplot(2,1,2);  plot([1:length(posnetek3D)]/Fs,posnetek3D(:,2),'r'); 
hold on; plot([1:length(posnetek)]/Fs,posnetek(:,2));
axis tight; title('kanal 2')
xlabel('�as (s)');
ylabel('amplituda');

sound(posnetek3D,Fs);

