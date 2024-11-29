from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager

# Define a Manager class
class QueueManager(BaseManager):
    pass

# Create the queues
response_ids_Q = Queue(maxsize=1)
prompt_Q = Queue(maxsize=1)

# Register the queues with the manager
QueueManager.register('get_response_ids_Q', callable=lambda: response_ids_Q)
QueueManager.register('get_prompt_Q', callable=lambda: prompt_Q)

def start_manager():
    manager = QueueManager(address=('', 50000), authkey=b'secret')
    server = manager.get_server()
    server.serve_forever()

# Start the manager server
manager_process = Process(target=start_manager)
manager_process.start()