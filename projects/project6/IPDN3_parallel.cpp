//
//  main.cpp
//  IPD++
//
//  Created by Arend Hintze on 2021-04-11.
//  Developed by Ali Eshtehardian on 2024-08-04

#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <memory>
#include <chrono>
#include <omp.h>

using namespace std;

mt19937 generator;

#define PI 3.14159265359
#define rand_uniform(lower,upper) uniform_real_distribution<double>(lower,upper)(generator)
#define randP uniform_real_distribution<double>(0.0,1.0)(generator)
#define rand_normal(mean,var) normal_distribution<double>(mean,var)(generator)
#define rand_index(lower,upper) uniform_int_distribution<int>(lower,upper)(generator);

//Here are the parameters that we are using for evolution purposes
int popSize=100;
int neighbours=20; //1/2 of the number of opponents
int gameLength=50;
double my=0.01;  //mutation rate
double nr=0.1;  //noise rate
int generations=10000;
string fileLead="";
int replicateNumber=0;
int popSaveIntervall=100;
double payoff[4]={4.0,0.5,8.0,1.0};//the payoff matrix
double strengthOfSelection[9]={5e-5, 5e-4, 5e-3,1e-2, 3e-2, 5e-2, 1e-1, 5e-1, 5};

const int anti_move[2]={0,1};

// Our agent has two genes, p[0] which is p(C|C) and p[1] which is p(C|D)
class Agent{
public:
    double p[2];
    double pi[2];
    double score;
    shared_ptr<Agent> ancestor;

    Agent(shared_ptr<Agent> from, double my);
    void play(shared_ptr<Agent> opponent, double nr);
};

//The function that describes the Shannon Entropy
double H(double p){

    if (p!=0.0 && p!=1.0){

    double output = -p*log2(p)-(1-p)*log2(1-p);
    return output;
    }else{
      return 0.0;
      }
}

int main(int argc, const char * argv[]) {

    auto start = std::chrono::high_resolution_clock::now();

for(int m=0;m<9;m++){

    cout<<"Strength of the Selection: "<<strengthOfSelection[m]<<endl;

    for(int r=0;r<5;r++){




    //Initialization of our arrays. I means information, hpic means entropy of cooperation and Ihpic means I/H
    double p1[generations];
    double p2[generations];
    double pi_c[generations];
    double pi_ce[generations];
    double I[generations];
    double Ie[generations];
    double hpic[generations];
    double Ihpic[generations];
    double Ihpice[generations];

    for(int k=0; k<popSize; ++k){

        p1[k]=0.0;
        p2[k]=0.0;
        pi_c[k]=0.0;
        pi_ce[k]=0.0;
        I[k]=0.0;
        hpic[k]=0.0;
    }

    vector<shared_ptr<Agent>> population,nextGen;

    //seed random number generator
    random_device actualRandomDevice;
    int theSeed = actualRandomDevice();
    generator.seed(theSeed);

    //This is the commandline parser:
    for(int i=1;i<argc;i+=2){
        if(string(argv[i]).compare("popsize")==0)
            popSize=atoi(argv[i+1]);
        if(string(argv[i]).compare("neighbours")==0)
            neighbours=atoi(argv[i+1]);
        if(string(argv[i]).compare("gameLength")==0)
            gameLength=atoi(argv[i+1]);
        if(string(argv[i]).compare("fileName")==0)
            fileLead=string(argv[i+1]);
        if(string(argv[i]).compare("generations")==0)
            generations=atoi(argv[i+1]);
//        if(string(argv[i]).compare("sos")==0)
//            strengthOfSelection=atof(argv[i+1]);
        if(string(argv[i]).compare("my")==0)
            my=atof(argv[i+1]);
        if(string(argv[i]).compare("ID")==0)
            replicateNumber=atoi(argv[i+1]);
        if(string(argv[i]).compare("intervall")==0)
            popSaveIntervall=atoi(argv[i+1]);
        if(string(argv[i]).compare("p0")==0)
            payoff[0]=atof(argv[i+1]);
        if(string(argv[i]).compare("p1")==0)
            payoff[1]=atof(argv[i+1]);
        if(string(argv[i]).compare("p2")==0)
            payoff[2]=atof(argv[i+1]);
        if(string(argv[i]).compare("p3")==0)
            payoff[3]=atof(argv[i+1]);
    }

    //initialize the population
    for(int i=0;i<popSize;i++){
        population.push_back(make_shared<Agent>(nullptr,0.0));
    }

    //run the whole thing
    for(int generation=0;generation<generations;generation++){

        //evaluate everyones score
        double meanPlay[]={0.0,0.0};

        //Parallelization which is explained in the report
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for(int n=0;n<popSize;n++){
            for(int k=0;k<neighbours;k++){
                population[n]->play(population[(n+k+1)%popSize], nr);
            }
        }

         //Parallelization which is explained in the report
        #pragma omp parallel for reduction(+:meanPlay[:2]) schedule(static)
        for(int n=0;n<popSize;n++){
                for(int i=0;i<2;i++){
                    meanPlay[i]+=population[n]->pi[i];
                }
        }

        double s=0.0;
        for(int i=0;i<2;i++){
            s+=meanPlay[i];
        }


        double x1=0.0;
        double x2=0.0;

        double x3=0.0;

        double x4=0.0;

        double x5=0.0;

        pi_c[generation]=meanPlay[0]/s;

         //Parallelization which is explained in the report
        #pragma omp parallel for reduction(+:x1, x2, x3, x4, x5) schedule(dynamic)

        for (int j=0; j<popSize; j++){

            x1+=population[j]->p[0];

            x2+=population[j]->p[1];

            x3+=H(pi_c[generation]*(population[j]->p[0]*(1-nr)+population[j]->p[1]*nr)+(1-pi_c[generation])*(population[j]->p[1]*(1-nr)+population[j]->p[0]*nr))+(-pi_c[generation]*H((population[j]->p[0])*(1-nr)+nr*(population[j]->p[1]))-(1-pi_c[generation])*H((population[j]->p[1])*(1-nr)+nr*(population[j]->p[0])));

            x4+=1-(pi_c[generation]*H(population[j]->p[0]*(1-nr)+population[j]->p[1]*nr)+(1-pi_c[generation])*H(population[j]->p[1]*(1-nr)+population[j]->p[0]*nr))/H(pi_c[generation]*(population[j]->p[0]*(1-nr)+population[j]->p[1]*nr)+(1-pi_c[generation])*(population[j]->p[1]*(1-nr)+population[j]->p[0]*nr));


        }

        p1[generation]=x1/popSize;
        p2[generation]=x2/popSize;

        pi_ce[generation]=pi_c[generation];
        hpic[generation]=H(pi_c[generation]);
        I[generation]=x3/popSize;


        Ie[generation]=H((1-nr)*(p1[generation]*pi_c[generation]+p2[generation]*(1-pi_c[generation]))+nr*(p2[generation]*pi_c[generation]+p1[generation]*(1-pi_c[generation])))-pi_c[generation]*H(p1[generation]*(1-nr)+nr*p2[generation])-(1-pi_c[generation])*H(p2[generation]*(1-nr)+nr*p1[generation]);

        Ihpic[generation] = x4/popSize;
        Ihpice[generation] = Ie[generation]/hpic[generation];

        //make selection and next population
        nextGen.clear();
        #pragma omp parallel for schedule(dynamic)
        for(int i=0;i<popSize;i++){
            int who=rand_index(0, popSize-1);
            int other=rand_index(0,popSize-1);
            while(who==other){
                who=rand_index(0, popSize-1);
                other=rand_index(0,popSize-1);
            }
            double p1=1.0/(1+exp((population[other]->score/(2*gameLength*neighbours)-population[who]->score/(2*gameLength*neighbours))/strengthOfSelection[m]));
            if(randP>p1){
                who=other;
            }
            #pragma omp critical
            nextGen.push_back(make_shared<Agent>(population[who],my));
        }

        population=nextGen;


    }


    int gencounter[generations];
    for (int j=0; j<generations; j++){

        // cout<<p1[j]<<endl;
        gencounter[j]=j;
    }


     ofstream file("p1_"+to_string(strengthOfSelection[m])+"_"+to_string(r)+".csv");

    if (file.is_open()) {
        // Assuming you are writing some x and y data points
        for (int i = 0; i < generations; ++i) {
            double x = gencounter[i]; // your data for x
            //  if(isnan(p1[i])){

            //     // cout<<p1<<endl;

            //     p1[i]=0.0;
            //     // cout<<p1<<endl;

            // }
            double y = p1[i]; // example data for y, e.g., y = x^2
            file << x << "," << y << "\n";
        }
        file.close();
    } else {
        cout << "Unable to open file";
    }


    ofstream file2("p2_"+to_string(strengthOfSelection[m])+"_"+to_string(r)+".csv");

    if (file2.is_open()) {
        // Assuming you are writing some x and y data points
        for (int i = 0; i < generations; ++i) {
            double x2 = gencounter[i]; // your data for x
            //   if(isnan(p2[i])){

            //     p2[i]=0.0;
            // }
            double y2 = p2[i]; // example data for y, e.g., y = x^2
            file2 << x2 << "," << y2 << "\n";
        }
        file2.close();
    } else {
        cout << "Unable to open file";
    }


      ofstream file3("pic_"+to_string(strengthOfSelection[m])+"_"+to_string(r)+".csv");

    if (file3.is_open()) {
        // Assuming you are writing some x and y data points
        for (int i = 0; i < generations; ++i) {
            double x2 = gencounter[i]; // your data for x
            //   if(isnan(p2[i])){

            //     p2[i]=0.0;
            // }
            double y2 = pi_c[i]; // example data for y, e.g., y = x^2
            file3 << x2 << "," << y2 << "\n";
        }
        file3.close();
    } else {
        cout << "Unable to open file";
    }


     ofstream file4("I_"+to_string(strengthOfSelection[m])+"_"+to_string(r)+".csv");

    if (file4.is_open()) {
        // Assuming you are writing some x and y data points
        for (int i = 0; i < generations; ++i) {
            double x2 = gencounter[i]; // your data for x
            //   if(isnan(p2[i])){

            //     p2[i]=0.0;
            // }
            double y2 = I[i]; // example data for y, e.g., y = x^2
            file4 << x2 << "," << y2 << "\n";
        }
        file4.close();
    } else {
        cout << "Unable to open file";
    }

     ofstream file5("hpic_"+to_string(strengthOfSelection[m])+"_"+to_string(r)+".csv");

    if (file5.is_open()) {
        // Assuming you are writing some x and y data points
        for (int i = 0; i < generations; ++i) {
            double x2 = gencounter[i]; // your data for x
            //   if(isnan(p2[i])){

            //     p2[i]=0.0;
            // }
            double y2 = hpic[i]; // example data for y, e.g., y = x^2
            file5 << x2 << "," << y2 << "\n";
        }
        file5.close();
    } else {
        cout << "Unable to open file";
    }


     ofstream file6("Ihpic_"+to_string(strengthOfSelection[m])+"_"+to_string(r)+".csv");

    if (file6.is_open()) {
        // Assuming you are writing some x and y data points
        for (int i = 0; i < generations; ++i) {
            double x2 = gencounter[i]; // your data for x
            //   if(isnan(p2[i])){

            //     p2[i]=0.0;
            // }
            double y2 = Ihpic[i]; // example data for y, e.g., y = x^2
            file6 << x2 << "," << y2 << "\n";
        }
        file6.close();
    } else {
        cout << "Unable to open file";
    }


        ofstream file7("Ie_"+to_string(strengthOfSelection[m])+"_"+to_string(r)+".csv");

        if (file7.is_open()) {
            // Assuming you are writing some x and y data points
            for (int i = 0; i < generations; ++i) {
                double x2 = gencounter[i]; // your data for x
                //   if(isnan(p2[i])){

                //     p2[i]=0.0;
                // }
                double y2 = Ie[i]; // example data for y, e.g., y = x^2
                file7 << x2 << "," << y2 << "\n";
            }
            file7.close();
        } else {
            cout << "Unable to open file";
        }

        ofstream file8("Ihpice_"+to_string(strengthOfSelection[m])+"_"+to_string(r)+".csv");

        if (file8.is_open()) {
            // Assuming you are writing some x and y data points
            for (int i = 0; i < generations; ++i) {
                double x2 = gencounter[i]; // your data for x
                //   if(isnan(p2[i])){

                //     p2[i]=0.0;
                // }
                double y2 = Ihpice[i]; // example data for y, e.g., y = x^2
                file8 << x2 << "," << y2 << "\n";
            }
            file8.close();
        } else {
            cout << "Unable to open file";
        }


   }

}



    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Total Execution Time: " << elapsed.count() << " seconds" << endl;

    ofstream file9("../Time/timing.csv", std::ios::app);  // append mode

    if (file9.is_open()) {

        file9 << elapsed.count() << "\n";

        file9.close();
    } else {
        cout << "Unable to open file";
    }




    return 0;
}


//In this function, our agents (players) are being selected for going to the next generation
Agent::Agent(shared_ptr<Agent> from,double my){
    ancestor=nullptr;
    score=0.0;
    if(from==nullptr){

                for(int i=0;i<2;i++){

                p[i]=randP;
                pi[i]=0.0;

                }




    } else {
        ancestor=from;
        for(int i=0;i<2;i++){
            pi[i]=0.0;
            if(randP<my)
                p[i]=randP;

            else
                p[i]=from->p[i];
        }
    }
}


//In this function, we have our game, where noise rate (nr) can flip the players last movement

void Agent::play(shared_ptr<Agent> opponent, double nr){
    //considering 0 as C and 1 as D
    int lastMove=rand_index(0,1);
    int lastMove2=rand_index(0,1);
    double avgscore=0.0;
    for(int r=0;r<gameLength;r++){

        if (randP<nr){

            lastMove=1-lastMove;


        }

        if (randP<nr){

            lastMove2=1-lastMove2;


        }

        int selfC=(randP<p[lastMove])?0:1;
        int otherC=(randP<opponent->p[lastMove2])?0:1;
        lastMove=otherC;
        lastMove2=selfC;

        if(selfC==0 && otherC==0){
        score+=payoff[0];
        opponent->score+=payoff[0];
        pi[0]+=1.0;
        opponent->pi[0]+=1.0;
        }

        if(selfC==0 && otherC==1){
        score+=payoff[1];
        opponent->score+=payoff[2];
        pi[0]+=1.0;
        opponent->pi[1]+=1.0;
        }

        if(selfC==1 && otherC==0){
        score+=payoff[2];
        opponent->score+=payoff[1];
        pi[1]+=1.0;
        opponent->pi[0]+=1.0;
        }

        if(selfC==1 && otherC==1){
        score+=payoff[3];
        opponent->score+=payoff[3];
        pi[1]+=1.0;
        opponent->pi[1]+=1.0;
        }
    }
}
