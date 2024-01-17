#include <bits/stdc++.h>
#include <limits.h>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define INF 1e9

int ans = INF;
int anss;
double costs[60 * 60];
vector<double> pheromones;
int n, m, t;

using namespace std;

const int MAX_CITIES = 100;
const double ALPHA = 1.0; // 影響信息素的重要程度
const double BETA = 2.0;  // 影響啟發信息的重要程度
const double RHO = 0.5;   // 信息素的蒸發率

struct Ant {
    vector<int> tour;  // 螞蟻遊覽路線
    double tourLength; // 螞蟻遊覽路線的總成本
};

vector<Ant> ants(10000);

void initPheromones(vector<double>& pheromones, int n) {
    // 初始化信息素矩陣
    pheromones.resize(n * n, 1.0);
}

double calculateProbability(int i, int j, const vector<double>& pheromones, int n, const double costs[]) {
    // 計算螞蟻從城市i移動到城市j的概率
    double numerator = pow(pheromones[i * n + j], ALPHA) * pow(1.0 / costs[i * n + j], BETA);
    double denominator = 0.0;

    for (int k = 0; k < n; ++k) {
        if (k != i) {
            denominator += pow(pheromones[i * n + k], ALPHA) * pow(1.0 / costs[i * n + k], BETA);
        }
    }

    return numerator / denominator;
}

int selectNextCity(int i, const vector<double>& pheromones, int n, const double costs[], const vector<bool>& visited) {
    // 根據概率選擇下一個城市
    double randValue = ((double)rand() / RAND_MAX);
    double sum = 0.0;

    for (int j = 0; j < n; ++j) {
        if (!visited[j]) {
            sum += calculateProbability(i, j, pheromones, n, costs);
            if (randValue <= sum) {
                return j;
            }
        }
    }

    // 如果所有城市都被訪問過，則選擇未訪問的城市中第一個城市
    for (int j = 0; j < n; ++j) {
        if (!visited[j]) {
            return j;
        }
    }

    return -1; // 應該永遠不會執行到這裡
}

void updatePheromones(vector<double>& pheromones, const vector<Ant>& ants, double evaporationRate, int n) {
    // 更新信息素矩陣
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            pheromones[i * n + j] *= (1.0 - evaporationRate); // 蒸發信息素

            for (int r=0;r<m;r++) {
                for (int k = 0; k < ants[r].tour.size() - 1; ++k) {
                    if ((ants[r].tour[k] == i && ants[r].tour[k + 1] == j) || (ants[r].tour[k] == j && ants[r].tour[k + 1] == i)) {
                        pheromones[i * n + j] += (1.0 / ants[r].tourLength); // 更新信息素
                        break;
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    srand(time(0));
    
    if(rank==0)
    {
        char input_file_name[10000];
        scanf("%s",input_file_name);
        ifstream myfile;
        myfile.open(input_file_name);
        if(!myfile.is_open())
        {
        	cerr<<"Error opening file:"<<input_file_name<<endl;
        	return 1;
        }
        
        myfile>>n>>m>>t;
    
        // 讀取城市之間的成本
        for (int i = 0; i < n * n; ++i) {
            for(int j=0;j<n;j++)
            {
                myfile >> costs[i*n+j];
            }
        }
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);

    m /= size;

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(costs, n * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }
    if (rank != 0) {
        MPI_Recv(costs, n * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 初始化信息素矩陣
    initPheromones(pheromones, n);

    for (int iteration = 0; iteration < t; iteration++) {

        // 每螞蟻找尋一條路徑
        for (int i = 0; i < m; ++i) {
            ants[i].tour.clear();
            ants[i].tourLength = 0.0;

            vector<bool> visited(n, false);
            int startCity = rand() % n;
            ants[i].tour.push_back(startCity);
            visited[startCity] = true;

            for (int j = 1; j < n; ++j) {
                int nextCity = selectNextCity(ants[i].tour.back(), pheromones, n, costs, visited);
                ants[i].tour.push_back(nextCity);
                visited[nextCity] = true;
                ants[i].tourLength += costs[ants[i].tour[j - 1] * n + nextCity];
            }

            // 回到起始城市
            ants[i].tour.push_back(startCity);
            ants[i].tourLength += costs[ants[i].tour.back() * n + startCity];
        }

        // 更新信息素
        updatePheromones(pheromones, ants, RHO, n);

        // 找出最短路徑
        double bestLength = ants[0].tourLength;
        int bestAnt = 0;
        #pragma omp parallel
        {
     	      #pragma omp for
            for (int i = 1; i < m; ++i) {
                if (ants[i].tourLength < bestLength) {
                    bestLength = ants[i].tourLength;
                    bestAnt = i;
                }
            }
        }
        #pragma omp barrier

        // 輸出最短路徑長度
        if(bestLength<ans)
        {
            ans=bestLength;
        }
    }
    
    MPI_Reduce(&ans, &anss, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    
    if(rank==0)
    {
        printf("%d",anss);
    }

    MPI_Finalize();
}
