class Reconfigurations:
    def __init__(self, nodes):
        self.nodes = nodes
        print("self node di init: {}".format(self.nodes))
        # for node in self.nodes:
        #     # print("node 0 : ", node["name"])
        #     if node.name == "Hls1":
        #         # cmdReconfiguration = "ls"
        #         # os.system(cmdReconfiguration)
        #         self.kill("Hls1")
        #         # node.stop()

    @staticmethod
    def create():
        print("reconfig class, create")

    @staticmethod
    def delete():
        print("reconfig class, delete")

    # @staticmethod
    def kill(self, node):
        print("reconfig class, kill {}".format(node))

    @staticmethod
    def modify():
        print("reconfig class, modify")

    @staticmethod
    def reroute():
        print("reconfig class, reroute")

    @staticmethod
    def rollback():
        print("reconfig class, rollback")

    @staticmethod
    def restart():
        print("reconfig class, restart")
