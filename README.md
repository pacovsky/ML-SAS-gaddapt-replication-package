# Replication Package
For paper Generalization of Machine-learning Adaptation in Ensemble-based Self-adaptive Systems

This repository holds code and data required to replicate our experiments.

All code files provided in this repository are experimental prototypes and should be regarded as such. They are provided AS IS with no warranties whatsoever.

## Run as
python en2-drone-charging/drone_charging_example/run.py  en2-drone-charging/drone_charging_example/experiments/FINAL/5dice-3.yaml -r 11111100

Important options that override the scenario behavior are:
	
	--seed
	--max_steps
	--drones
	--birds
	
For concrete method choose accordingly from following options:

	--hysteresis either use 'd0.15-10' or do not use the option at all
	-r is sanitation use one of 00000000, 11111100, turing all sanitations off or on.
	--which_simulation select one of [betterbaseline, argmax, greedy_drone, greedy_ensemble_priority, slowProb, slowPoints ] these names correspond to the paper names [baseline, plain neural network, greedy drone first, greedy ensemble first, optimal method with Value metrics, optimal method with Order metrics]

[//]: # (-r -sanitation - use 00000000 or 11111100 -r = subfolder - RF)
[//]: # (Todo add '--sanitations' as True / False instead)



## Maps detail description

The maps used in the paper with the respective filenames and a map description.

| **Name**      | **Chargers** | **Drones** | **Birds** | **Map Size** | **Need** | **Food** | **Field Density** | **B-D Ratio** | **Need-Ratio** | **FUpB** | **FUpD** | **Most Common Category** | **Size** | **Drones** | **Density** | **Food** | **Attackers** | **Protectors** | **Components Count** | **Food Density** | **Triplet** | **Decription** |
|---------------|--------------|------------|-----------|--------------|----------|----------|-------------------|---------------|----------------|----------|----------|--------------------------|----------|------------|-------------|----------|---------------|----------------|----------------------|------------------|-------------|----------------|
| **Real-Life** | 4            | 120        | 100       | 3312         | 60       | 1003     | 0.3               | 1.2           | 2              | 10       | 8.33     | L                        | M        | L          | L           | L        | L             | L              | L                    | L                | L-L-4       | L,4ch,>100d\   |
| **U-1-Ch**    | 1            | 24         | 32        | 5100         | 20       | 925      | 0.18              | 0.75          | 0.83           | 29       | 38.54    | M                        | L        | M          | M           | L        | M             | M              | M                    | M                | M-M-1       | M,1ch,24d,32b  |
| **Fill**      | 1            | 30         | 50        | 2500         | 23       | 985      | 0.4               | 0.6           | 0.77           | 20       | 32.83    | M                        | S        | M          | L           | L        | M             | M              | M                    | L                | M-L-1       | M,4ch,30d,50b  |
| **5dice-3**   | 4            | 15         | 30        | 2400         | 5        | 442      | 0.18              | 0.5           | 3              | 15       | 28.13    | M                        | S        | L          | M           | M        | L             | M              | M                    | M                | M-M-4       | M,1ch,15d,30b  |
| **4d\_lr**    | 1            | 4          | 5         | 3312         | 10       | 233      | 0.07              | 0.8           | 0.4            | 47       | 58.25    | S                        | M        | S          | S           | S        | S             | S              | S                    | S                | S-S-1       | S,4ch,4d,5b    |
| **10drones**  | 3            | 15         | 30        | 2500         | 26       | 942      | 0.38              |               |                |          |          |                          |          |            |             |          |               |                |                      |                  |             |                |


   
Need= Drones Need To Fully Cover All Fields, 

FUPD=Field units per drone, 

FUpB Food units per bird.
