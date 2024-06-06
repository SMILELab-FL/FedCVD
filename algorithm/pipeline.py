
from fedlab.core.standalone import StandalonePipeline


class Pipeline(StandalonePipeline):
    def main(self):
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            # evaluate
            self.evaluate()
            self.handler.round += 1
            self.trainer.current_round = self.handler.round
        self.trainer.finish()
        self.handler.finish()
        # for idx in range(self.trainer.num_clients):
        #     self.trainer.evaluators[idx].save(self.trainer.output_path + "client" + str(idx + 1) + "/metric.json")
        # self.handler.evaluator.save(self.handler.output_path + "server/metric.json")
        # self.handler.save_model(self.handler.output_path + "server/model.pth")

    def evaluate(self):
        self.handler.local_test()
        self.handler.global_test()
