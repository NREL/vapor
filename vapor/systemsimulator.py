import logging
log = logging.getLogger("vapor")

from vapor.systemsimulator_objects import *

class MeetGoalMerchantPlant(GenericSystemSimulator, MeetGoalAddon, BayesianSimulatorAddon):
    def simulate(self):
        self.optimize()

class FixedCapacityMerchantPlant(GenericSystemSimulator, FixedCapacityAddon, BayesianSimulatorAddon):
    def simulate(self):
        self.optimize()

class FixedCapacitySelfConsumption(GenericSystemSimulator, FixedCapacityAddon, BayesianSimulatorAddon):
    def simulate(self):
        self.optimize()
