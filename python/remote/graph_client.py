import numpy as np
import asyncio
from remote.bytes_to_graph import bytes_to_graph
from remote.graph_to_bytes import graph_to_bytes

class GraphClient:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.writer = None
        self.reader = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

    async def optimize(self, graph):
        print("optimize")

        if self.writer is None:
            return

        self.writer.write(graph_to_bytes(graph))
        await self.writer.drain()

        result = await self.wait_for_graph()

        return result

    async def wait_for_graph(self):

        print("wait_for_graph")
        while True:
            if self.reader is None:
                return
                
            data_size_bytes = await self.reader.read(4)
            if not data_size_bytes:
                return
            data_size = int(np.frombuffer(data_size_bytes[:4], dtype=np.uint32)[0])
            print(f"graph data_size: {data_size}")
            
            graph = None
            
            graph_bytes = b''
            while len(graph_bytes) < data_size:
                packet = await self.reader.read(data_size - len(graph_bytes))
                if not packet:
                    raise ValueError("Connection closed by the server")
                graph_bytes += packet
            graph = bytes_to_graph(graph_bytes)

            return graph
    
    async def close(self):
        if self.writer is None:
            return
        
        self.writer.close()
        await self.writer.wait_closed()