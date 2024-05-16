from datetime import datetime
import time
from enum import Enum
import pandas as pd
from memory_profiler import memory_usage
import os.path

class Criteria(Enum):
    FUNCTIONAL = 0
    RUNTIME = 1
    CPU_RUNTIME = 2
    STORAGE = 3

class Benchmark:
    # Class attributes
    runtime_avg  = 0
    runtime_max = 0
    runtime_cpu_avg  = 0
    runtime_cpu_max  = 0
    ram_avg = 0
    ram_max = 0

    def __init__(self, tool, dataset, criterias , buildings, vm):
        self.tool = tool
        self.dataset = dataset
        self.criterias = criterias
        self.buildings = buildings
        self.runOnVm = vm
        self.driftDetectionStats = {}

    def runBenchmark(self):
        for criteria in self.criterias:
            if criteria == Criteria.FUNCTIONAL:
                self.runFunctional()
            elif criteria == Criteria.RUNTIME:
                self.runRuntime()
            elif criteria == Criteria.CPU_RUNTIME:
                self.runCPUruntime()
            elif criteria == Criteria.STORAGE:
                self.runStorage()

        # generate Report
        self.__printReport()

    def __printReport(self):
        self.driftDetectionStats=pd.DataFrame.from_dict(self.driftDetectionStats)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("==============================")
        name = self.tool.name
        if self.tool.showReport:
            name = name + " with report"
        print("Benchmark Report: {}".format(name))
        print("Report generated at: {}".format(current_time))
        print("==============================")
        for x in self.driftDetectionStats:
            print("Gebäude {}".format(x))
            building_stats = self.driftDetectionStats[x]
            column_names = self.tool.column_names
            
            # Extract the drift scores and is_drifted values
            for col in column_names:
                drift_score = {test: stats.get(f"{col}_drift_score", stats.get("drift_score"))
                                for test, stats in building_stats.items()
                                if isinstance(stats, dict)}
                is_drifted = {test: stats.get(f"{col}_is_drifted", stats.get("is_drifted"))
                                for test, stats in building_stats.items()
                                if isinstance(stats, dict)}
                print(f"Column: {col}")
                print(f"Drift Score: {drift_score}")
                print(f"Is Drifted: {is_drifted}")
                print()

            print()
        print("==============================")
        print("Runtime AVG: {:.8f} milliseconds".format(self.runtime_avg))
        print("Runtime MAX: {:.8f} milliseconds".format(self.runtime_max))
        print("CPU Runtime AVG: {:.7f} milliseconds".format(self.runtime_cpu_avg))
        print("CPU Runtime MAX: {:.7f} milliseconds".format(self.runtime_cpu_max))
        print("RAM Usage AVG: {:.7f} MiB".format(self.ram_avg))
        print("RAM Usage MAX: {:.7f} MiB".format(self.ram_max))
        print("==============================")

        self.__saveReport(current_time=current_time)

    
    def __saveReport(self, current_time):
        self.driftDetectionStats = pd.DataFrame.from_dict(self.driftDetectionStats)
        column_names = self.tool.column_names

        # Append data for each test to the report list
        report_data = []

        for x in self.driftDetectionStats:
            building_stats = self.driftDetectionStats[x]

            # Extract the test names
            tests = [test for test, stats in building_stats.items() if isinstance(stats, dict)]

            for test in tests:
                stats = building_stats[test]

                # Create a dictionary to hold the row's data
                row_data = {
                    'time': current_time,
                    'building': x,
                    'tool': self.tool.name,
                    'showReport': self.tool.showReport,
                    'test': test,
                    'runtime_avg': self.runtime_avg,
                    'runtime_max': self.runtime_max,
                    'cpu_runtime_avg': self.runtime_cpu_avg,
                    'cpu_runtime_max': self.runtime_cpu_max,
                    'ram_avg': self.ram_avg,
                    'ram_max': self.ram_max,
                    'run_on_vm': self.runOnVm,
                }

                for col in column_names:
                    col_drift_score = f"{col}_drift_score"
                    col_is_drifted = f"{col}_is_drifted"

                    drift_score = stats.get(col_drift_score)
                    is_drifted = stats.get(col_is_drifted)

                    if (drift_score == None) & (is_drifted == None):
                        row_data['all_col_drift_score'] = stats.get('drift_score')
                        row_data['all_col_is_drifted'] = stats.get('is_drifted')
                    else:
                        row_data[col_drift_score] = drift_score
                        row_data[col_is_drifted] = is_drifted

                report_data.append(row_data)

        # Create a DataFrame from the report data
        report_df = pd.DataFrame(report_data)

        if os.path.exists('benchmark_report.csv'):
            report_df.to_csv('benchmark_report.csv', mode='a', index=False, header=False)
        else:
            report_df.to_csv('benchmark_report.csv', index=False)

    def runFunctional(self):
        for building_id in self.buildings:
            # Split into reference and current dataset
            ref, cur = self.dataset.splitTrainTest(building_id)
            self.driftDetectionStats[building_id] = {}

            # runDriftDetection without report generation
            my_dict = self.tool.runDriftdetection(ref, cur, building_id)
            self.ref = ref
            if not my_dict:
                print("Dict from building {} is empty" .format(building_id))
            else:
                self.driftDetectionStats[building_id].update(my_dict)

    # measures elapsed time using wall-clock time (include: waiting time for resources, dependant on other processes): time in ms
    def runRuntime(self):  
        runtime_sum = 0  
        self.runtime_max = 0

        for building_id in self.buildings:
            # Split into reference and current dataset
            ref, cur = self.dataset.splitTrainTest(building_id)

            # Timestamp before executing drift detection
            st = time.time()
            self.tool.runDriftdetection(ref, cur, building_id)

            # Timestamp after executing, compute runtime in ms, divided by count of algorithms of tool, result: avg runtime of 1 algorithm for 1 building
            runtime_result = ((time.time() - st)/len(self.tool.methods))  * 1000 
            runtime_sum += runtime_result

            # set maximal runtime
            self.runtime_max = max(self.runtime_max, runtime_result)

        # compute average runtime
        self.runtime_avg = runtime_sum / len(self.buildings) 

    # measures cpu resources (user and system) consumed by the process (exclude: waiting time for resources): time in ms
    def runCPUruntime(self):
        cpu_sum = 0  
        self.runtime_cpu_max = 0
        for building_id in self.buildings:
            ref, cur = self.dataset.splitTrainTest(building_id)

            # Start measuring CPU Usage (include user and system cpu time)
            st = time.process_time() 
            self.tool.runDriftdetection(ref, cur, building_id)

            # End measuring CPU Usage, compute cpu in GB
            end = time.process_time()
            cpu_result = ((end-st) / len(self.tool.methods)) * 1000
            cpu_sum += cpu_result

            # set maximal cpu
            self.runtime_cpu_max = max(self.runtime_cpu_max, cpu_result)

        # compute average cpu
        self.runtime_cpu_avg = cpu_sum / len(self.buildings) 

    def runStorage(self):
        self.ram_max = 0
        mem = []
        for building_id in self.buildings:
            ref, cur = self.dataset.splitTrainTest(building_id)
            mem = mem + memory_usage((self.tool.runDriftdetection, (ref,cur,building_id)))

        # compute average storage
        self.ram_avg = sum(mem) / len(mem)
        self.ram_max = max(mem)