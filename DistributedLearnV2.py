## below code is expected to start 2 processes on different gpu's and with a parameter server placed in the cpu.
###### The code trains the neural network with 2 hidden layers of size 4096 and ann output layer of 256

import multiprocessing
import time
import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import psutil
import numpy as np
from tensorflow.python.client import device_lib

cluster_spec ={"ps": ["localhost:2223"
                      ],
    "worker": [
        "localhost:2224",
        "localhost:2225"]}

cluster = tf.train.ClusterSpec(cluster_spec)
#loss = tf.random_normal([1],dtype=tf.float32)
#print(cluster.task_address())
#X = tf.constant(np.random.random_sample(size=(128,256)), dtype=tf.float32)
#labls=tf.constant([1,0,0,1,0,1,0,1,0,1],shape=[10,1],dtype=tf.float32)

def dumb_input_fn():

    x = tf.random_normal([128,256], dtype=tf.float32)
    y = tf.random_normal([128,256], dtype=tf.float32)

    return [x,y]

output_dir = 'tfprojects/output_dir_debug'
#tf.logging.set_verbosity(tf.logging.INFO)  # enables training error print out during training
def model_fn(features,labels):
    # This is is reeeeally slow way to square numbers.
    outputs = tf.layers.dense(
                    inputs = features,
                    units = 4096)
    outputs = tf.layers.dense(
                    inputs = outputs,
                    units = 4096)
    outputs = tf.layers.dense(
                    inputs = outputs,
                    units = 256)

    loss = tf.losses.mean_squared_error(outputs, labels)


    train_op = tf.contrib.layers.optimize_loss(
              loss, None, optimizer='Adam',
                        learning_rate = .0001)


    predictions = {"predictions":tf.identity(outputs,name = 'predictions')}
    return predictions, loss, train_op



def worker(device):
    params = HParams(cluster=cluster,
                     job_name = device[0],
                     task_index = device[1])

    if device[0]=='worker':
        # allow each worker to see only 1 of the 4 GPUS
        os.environ["CUDA_VISIBLE_DEVICES"]=str(params.task_index)

    else:
        # hide all 4 GPUS from ps
        os.environ["CUDA_VISIBLE_DEVICES"]=''


    Workrs(output_dir, params)

class _LoggerHook():
    ## create before run and after run functions that could be used to call certain parameters
    def begin(self):
        self._time=time.time()

    def before_run(self, run_context):
        self._time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):
        self._time = time.time()
        duration = time.time() - self._start_time
        loss_value = run_values.results


        format_str = ('loss = %.2f')
        print (format_str % (loss_value))


## The Workrs processes is triggered from the main function. T
def Workrs(output_dir,d={}):
    cluster = d.cluster
    job_name = d.job_name
    task_index = d.task_index
    # tf.train.Server creates a server cluster with the local host address as mentioned in cluster_Spec in tensorflow
    server= tf.train.Server(d.cluster,job_name = d.job_name,task_index=d.task_index)
    print("Task IndeX:",d.task_index)

    if job_name == "ps":
        ##try and wait for all the wokers to finish their tasks.
        ## Tensorflow does not stop the server process on PS to stop. This needs to be manually done by the code
        ## Have to setup a Tensorflow queue which is written to by Worker process once they finish training and read by PS to stop its session from running.
        server.join()
    elif job_name == "worker":
        #tf.train.replica_device_setter is used to automatically assign available devices for the specific task index
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/replica:0/task:%d" % task_index,
                cluster=cluster)):

            X, labls = dumb_input_fn()
            _,_,train_op=model_fn(X,labls,None,d)

        global_step = tf.contrib.framework.get_or_create_global_step()
        step = 0
        start_time = time.time()
        # These are hooks that are to be used to stop training after a certain global step value has been reached.
        hooks = tf.train.StopAtStepHook(last_step=10)
        # MonitoredTrainingSession sets up the tensorflow threads and queues.
        with tf.train.MonitoredTrainingSession(master=server.target,config=tf.ConfigProto(log_device_placement=True),
                                                   is_chief=(d.task_index == 0),
                                                   checkpoint_dir=output_dir,
                                                   hooks=[hooks]) as mon_sess:

            #ckpt = tf.train.get_checkpoint_state(output_dir)

            while not mon_sess.should_stop():

                _ = mon_sess.run(train_op)
                if step % 5 == 0:
                    print("Step:", step, 10 / (time.time() - start_time), 'steps/sec')
                    print("Proc:",)
                    start_time = time.time()

                step += 1

#def get_available_gpus():
#    local_device_protos = device_lib.list_local_devices()
#    return [x.name for x in local_device_protos if x.device_type == 'GPU']
## Test the multiprocessing on a single cpu vs 4 on the local machine
#print("There are %d CPUs on this machine" % multiprocessing.cpu_count())
#print("There are  CPUs on this machine:" , get_available_gpus())
#pool = multiprocessing.Pool(processes=4)
#start_time = time.time()
#results = [pool.apply_async(cube, args=(x,)) for x in range(17,25)]
#out = [p.get() for p in results]
#print(out)
#start_time = time.time()
#pool.close()
#pool.join()

if __name__ == '__main__':

    logger = multiprocessing.log_to_stderr()
    result = multiprocessing.Queue()
    processes = []
    devices = [['ps', 0],
               ['worker', 0],
               ['worker', 1],
               ]

    for i in (devices):
        start_time = time.time()
        proc = multiprocessing.Process(target=worker,args=(i,))
        processes.append(proc)
        proc.start()

    for p in processes:
        p.join()

    print("time taken = %d" % (start_time - time.time()))



    results = [result.get() for p in processes]
#process_names = [proc.get_cpu_times() for proc in psutil.Process()]



