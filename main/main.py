import Benchmark
from Benchmark import Criteria
from Tool import Evidently
from Tool import AlibiDetect
from Tool import NannyML
import Dataset
from Dataset import Data_Occupacy
from Dataset import Data_Energy
import os

def main():
    clean()
    ####################################
    # use can change or define the benchmark here:
    # 1. select dataset: True if you want to investigate energy dataset, False if you want to investigate Occupacy dataset
    energy = False 

    # 2. select the tools
    tools = {Evidently("Evidently", False), Evidently("Evidently", True),
              NannyML("NannyML", False), NannyML("NannyML", True), 
              AlibiDetect("AlibiDetect")} 

    # 3. select criteria
    criteria = [Criteria.FUNCTIONAL, Criteria.RUNTIME, Criteria.CPU_RUNTIME, Criteria.STORAGE]

    # 4. select if run on vm: True if run on vm, False if run locally
    vm = False

    # finished
    #####################################

    if energy:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'energy_data.csv')
        dataset= Data_Energy(path)

    else:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'occupacy_data.csv')
        dataset = Data_Occupacy(path)

    runBenchmark(buildings={1}, tests=criteria, tools=tools, vm = vm, dataset=dataset)
    print("---------Benchmark execution finished---------")

# one benchmark execution with given criteria, tools and dataset
def runBenchmark(buildings = {1}, tests=[Criteria.FUNCTIONAL, Criteria.RUNTIME, Criteria.CPU_RUNTIME, Criteria.STORAGE],
                  tools={(Evidently("Evidently", showReport=False))}, vm = False, 
                  dataset=Data_Energy(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'energy_data.csv'))):
    for tool in tools:
        benchmark = Benchmark.Benchmark(tool, dataset, tests, buildings, vm)
        benchmark.runBenchmark()

# delete all reports
def clean():
    tests = {'kolmogorov_smirnov', 'anderson', 'cramer_von_mises', 'ed', 'es', 'hellinger', 'jensenshannon', 
             'kl_div', 'mannw', 'psi', 't_test', 'wasserstein', 'jensen_shannon'}
    for i in range(1, 37):
        for test in tests:
            file_name = "evidently_report_{}_{}.html".format(i, test)
            if os.path.exists(file_name):
                os.remove(file_name)
            file_name = "nannyml_report_dist_{}_{}.svg".format(i, test)
            if os.path.exists(file_name):
                os.remove(file_name)
            file_name = "nannyml_report_drift_{}_{}.svg".format(i, test)
            if os.path.exists(file_name):
                os.remove(file_name)

if __name__ == "__main__":
    main()