from vrp_simulator import simulator
import json
import matplotlib.pyplot as plt

def main():
    #run the simulator with different k values and save the results into a json file
    #set the m as the reconducting times
    k = 10
    n = 20
    results = {}
    for i in range (1, k + 1):
        results[str(i)] = []
        for j in range(n):
            print(f"Simulating with k = {i} for the {j+1}th time...")
            results[str(i)].append(simulator(k=i))
    
    with open('results.json', 'w') as f:
        json.dump(results, f)
    
    average_results = {}
    for i in range(1, k + 1):
        average_results[str(i)] = sum(results[str(i)])/len(results[str(i)])

    #plot the results and save the plot into a pdf file
    plt.plot(average_results.keys(), average_results.values())
    plt.xlabel('k')
    plt.ylabel('Whole vehicles distance')
    plt.title('Whole vehicles distance vs. k')
    plt.savefig('whole_vehicles_distance_vs_k.pdf')


if __name__ == "__main__":
    main()