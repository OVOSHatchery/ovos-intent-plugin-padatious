from os.path import join, expanduser
from threading import Lock

from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home
from padatious import IntentContainer
from ovos_plugin_manager.intents import IntentExtractor, IntentPriority, IntentDeterminationStrategy


class PadatiousExtractor(IntentExtractor):
    keyword_based = False

    def __init__(self, config=None,
                 strategy=IntentDeterminationStrategy.SEGMENT_REMAINDER,
                 priority=IntentPriority.MEDIUM_HIGH,
                 segmenter=None):
        super().__init__(config, strategy=strategy,
                         priority=priority, segmenter=segmenter)
        data_dir = expanduser(self.config.get("data_dir", xdg_data_home()))
        cache_dir = join(data_dir, "padatious")
        self.lock = Lock()
        self.container = IntentContainer(cache_dir)

    def detach_intent(self, intent_name):
        if intent_name in self.registered_intents:
            LOG.debug("Detaching padatious intent: " + intent_name)
            with self.lock:
                self.container.remove_intent(intent_name)
        super().detach_intent(intent_name)

    def register_entity(self, entity_name, samples=None, reload_cache=True):
        samples = samples or [entity_name]
        super().register_entity(entity_name, samples)
        with self.lock:
            self.container.add_entity(entity_name, samples,
                                      reload_cache=reload_cache)

    def register_intent(self, intent_name, samples=None, reload_cache=True):
        samples = samples or [intent_name]
        super().register_intent(intent_name, samples)
        with self.lock:
            self.container.add_intent(intent_name, samples,
                                      reload_cache=reload_cache)
        self.registered_intents.append(intent_name)

    def register_entity_from_file(self, entity_name, file_name,
                                  reload_cache=True):
        super().register_entity_from_file(entity_name, file_name)
        with self.lock:
            self.container.load_entity(entity_name, file_name,
                                       reload_cache=reload_cache)

    def register_intent_from_file(self, intent_name, file_name,
                                  single_thread=True, timeout=120,
                                  reload_cache=True, force_training=True):
        super().register_intent_from_file(intent_name, file_name)
        try:
            with self.lock:
                self.container.load_intent(intent_name, file_name,
                                           reload_cache=reload_cache)
            self.registered_intents.append(intent_name)
            success = self._train(single_thread=single_thread,
                                  timeout=timeout,
                                  force_training=force_training)
            if success:
                LOG.debug(file_name + " trained successfully")
            else:
                LOG.error(file_name + " FAILED TO TRAIN")

        except Exception as e:
            LOG.exception(e)

    def _get_remainder(self, intent, utterance):
        if intent["name"] in self.intent_samples:
            return self.get_utterance_remainder(
                utterance, samples=self.intent_samples[intent["name"]])
        return utterance

    def calc_intent(self, utterance, min_conf=0.0):
        min_conf = min_conf or self.config.get("padatious_min_conf", 0.35)
        utterance = utterance.strip().lower()
        with self.lock:
            intent = self.container.calc_intent(utterance).__dict__
        if intent["conf"] < min_conf:
            return {"intent_type": "unknown", "entities": {}, "conf": 0,
                    "intent_engine": "padatious",
                    "utterance": utterance, "utterance_remainder": utterance}
        intent["utterance_remainder"] = self._get_remainder(intent, utterance)
        intent["entities"] = intent.pop("matches")
        intent["intent_engine"] = "padatious"
        intent["intent_type"] = intent.pop("name")
        intent["utterance"] = intent.pop("sent")

        if isinstance(intent["utterance"], list):
            intent["utterance"] = " ".join(intent["utterance"])
        return intent

    def _train(self, single_thread=True, timeout=120, force_training=True):
        with self.lock:
            return self.container.train(single_thread=single_thread,
                                        timeout=timeout,
                                        force=force_training,
                                        debug=True)
