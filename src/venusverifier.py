from multiprocessing import Process
import queue
from timeit import default_timer as timer

from sys import platform

if platform == "darwin":
    from src.queuewrapper import QueueWrapper as Queue
else:
    from multiprocessing import Queue

from src.layersmodel import LayersModel
from src.splittingprocess import SplittingProcess


class WorkerVerificationProcess(Process):
    TIMEOUT = 3600

    def __init__(self, id, jobs_queue, reporting_queue, print=False):
        super(WorkerVerificationProcess, self).__init__()

        self.id = id
        self.jobs_queue = jobs_queue
        self.reporting_queue = reporting_queue

        self.PRINT_TO_CONSOLE = print

    def run(self):
        while True:
            try:
                job_id, vmodel = self.jobs_queue.get(timeout=self.TIMEOUT)

                start = timer()
                vmodel.encode()
                (res, gap) = vmodel.verify()
                end = timer()
                runtime = end - start

                if self.PRINT_TO_CONSOLE:
                    print(
                        "Subprocess",
                        self.id,
                        "finished job",
                        job_id,
                        "result:",
                        res,
                        "in",
                        runtime,
                    )
                self.reporting_queue.put((job_id, res, runtime, gap))

            except queue.Empty:
                # to handle the case when the main process got killed,
                # but the workers remained alive.
                # the worker waits for at most 2 minutes to get a new job,
                # if no job was found, it terminates
                break


class VenusVerifier:
    def __init__(
        self, nmodel, spec, encoder_params, splitting_params, print_to_console=False
    ):
        super(VenusVerifier, self).__init__()

        self.TIME_LIMIT = encoder_params.TIME_LIMIT

        self.PARALLEL_PROCESSES_NUMBER = encoder_params.WORKERS_NUMBER

        # convert keras nmodel into our internal representation LayersModel
        lmodel = LayersModel(encoder_params.ENCODING)
        lmodel.load(nmodel, spec)

        # the queue to which all worker processes report the results
        # and the splitting process will store the total number of splits
        self.reporting_queue = Queue()

        jobs_queue = Queue()

        # compute the initial splits
        aux_splitting_process = SplittingProcess(
            0,
            lmodel,
            spec,
            splitting_params,
            encoder_params,
            jobs_queue,
            self.reporting_queue,
            print_to_console,
        )
        initial_splits = aux_splitting_process.get_initial_splits()
        # start a splitting process for each initial split
        self.splitting_processes = [
            SplittingProcess(
                i + 1,
                initial_splits[i][0],
                initial_splits[i][1],
                splitting_params,
                encoder_params,
                jobs_queue,
                self.reporting_queue,
                print_to_console,
            )
            for i in range(len(initial_splits))
        ]

        self.worker_processes = [
            WorkerVerificationProcess(
                i + 1, jobs_queue, self.reporting_queue, print_to_console
            )
            for i in range(self.PARALLEL_PROCESSES_NUMBER)
        ]

        self.PRINT_TO_CONSOLE = print_to_console

    def verify(self):

        start = timer()

        # start the splitting and worker processes
        for proc in self.splitting_processes:
            proc.start()
        for proc in self.worker_processes:
            proc.start()

        timedout_jobs_count = 0
        finished_jobs_count = 0
        finished_splitting_processes_count = 0

        result = None
        total_number_of_splits = -1

        """ 
        Read results from the reporting queue
        until encountered a True result, or
        until all the splits have completed
        """
        while True:
            try:
                job_id, res, runtime, extra = self.reporting_queue.get(
                    timeout=self.TIME_LIMIT - (timer() - start)
                )

                if res == True:
                    if self.PRINT_TO_CONSOLE:
                        print("Main process: read True. Terminating...")
                    result = ("True", "{}".format(finished_jobs_count + 1), extra)
                    break

                elif res == False:
                    finished_jobs_count += 1

                elif res == "Timeout":
                    finished_jobs_count += 1
                    timedout_jobs_count += 1

                elif res == SplittingProcess.TOTAL_JOBS_NUMBER_STRING:
                    # update the total_number of splits
                    if total_number_of_splits == -1:
                        total_number_of_splits = job_id
                    else:
                        total_number_of_splits += job_id
                    finished_splitting_processes_count += 1
                else:
                    raise Exception("Unexpected result read from reporting queue", res)

                # stopping conditions
                if (
                    total_number_of_splits != -1
                    and finished_splitting_processes_count
                    == len(self.splitting_processes)
                    and finished_jobs_count >= total_number_of_splits
                ):
                    if self.PRINT_TO_CONSOLE:
                        print(
                            "Main process: all subproblems have finished. Terminating..."
                        )
                    if timedout_jobs_count == 0:
                        result = ("False", total_number_of_splits, None)
                    else:
                        result = ("Timeout", total_number_of_splits, None)
                    break

            except queue.Empty:
                # Timout occured
                result = ("Timeout", finished_jobs_count, None)
                break
            except KeyboardInterrupt:
                # Received terminating signal
                result = ("Interrupted", finished_jobs_count, None)
                break

        """
        Terminate the splitting and worker processes.
        Especially relevant if there was one early True result.
        """
        try:
            for proc in self.splitting_processes:
                proc.terminate()
            for proc in self.worker_processes:
                proc.terminate()
        except:
            print("Error while attempting to terminate processes")

        return result
