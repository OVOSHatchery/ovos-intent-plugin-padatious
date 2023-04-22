from os.path import join, expanduser
from threading import Lock

from ovos_plugin_manager.templates.intents import IntentExtractor, IntentMatch, IntentPriority, \
    IntentDeterminationStrategy
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home
from padatious import IntentContainer


def _munge(name, skill_id):
    return f"{name}:{skill_id}"


def _unmunge(munged):
    return munged.split(":", 2)


class PadatiousExtractor(IntentExtractor):
    keyword_based = False

    def __init__(self, config=None,
                 strategy=IntentDeterminationStrategy.SEGMENT_REMAINDER,
                 priority=IntentPriority.MEDIUM_HIGH,
                 segmenter=None):
        super().__init__(config, strategy=strategy,
                         priority=priority, segmenter=segmenter)
        data_dir = expanduser(self.config.get("data_dir", xdg_data_home()))
        self.cache_dir = join(data_dir, "padatious")
        self.lock = Lock()
        self.engines = {}  # lang: IntentContainer

    def _get_engine(self, lang=None):
        lang = lang or self.lang
        if lang not in self.engines:
            self.engines[lang] = IntentContainer(self.cache_dir)
        return self.engines[lang]

    def detach_intent(self, skill_id, intent_name):
        for intent in self.registered_intents:
            if intent.name != intent_name or intent.skill_id != skill_id:
                continue
            LOG.debug("Detaching padatious intent: " + intent_name)
            with self.lock:
                for lang in self.engines:
                    self.engines[lang].remove_intent(_munge(intent.name,
                                                            intent.skill_id))
        super().detach_intent(intent_name)

    def register_entity(self, skill_id, entity_name, samples=None, lang=None, reload_cache=True):
        lang = lang or self.lang
        super().register_entity(skill_id, entity_name, samples, lang)
        container = self._get_engine(lang)
        samples = samples or [entity_name]
        with self.lock:
            container.add_entity(entity_name, samples, reload_cache=reload_cache)

    def register_intent(self, skill_id, intent_name, samples=None, lang=None, reload_cache=True):
        lang = lang or self.lang
        super().register_intent(skill_id, intent_name, samples, lang)
        container = self._get_engine(lang)
        samples = samples or [intent_name]
        intent_name = _munge(intent_name, skill_id)
        with self.lock:
            container.add_intent(intent_name, samples,
                                 reload_cache=reload_cache)

    def register_entity_from_file(self, skill_id, entity_name, file_name, lang=None, reload_cache=True):
        lang = lang or self.lang
        container = self._get_engine(lang)
        super().register_entity_from_file(skill_id, entity_name, file_name, lang)
        with self.lock:
            container.load_entity(entity_name, file_name,
                                  reload_cache=reload_cache)

    def register_intent_from_file(self, skill_id, intent_name, file_name, lang=None,
                                  single_thread=True, timeout=120,
                                  reload_cache=True, force_training=True):
        lang = lang or self.lang
        container = self._get_engine(lang)
        super().register_intent_from_file(skill_id, intent_name, file_name, lang)
        intent_name = _munge(intent_name, skill_id)
        try:
            with self.lock:
                container.load_intent(intent_name, file_name,
                                      reload_cache=reload_cache)
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

    def calc_intent(self, utterance, min_conf=0.0, lang=None, session=None):
        lang = lang or self.lang
        container = self._get_engine(lang)
        min_conf = min_conf or self.config.get("padatious_min_conf", 0.35)
        utterance = utterance.strip().lower()
        with self.lock:
            intent = container.calc_intent(utterance).__dict__
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

        intent_type, skill_id = _unmunge(intent["intent_type"])
        return IntentMatch(intent_service=intent["intent_engine"],
                           intent_type=intent_type,
                           intent_data=intent,
                           confidence=intent["conf"],
                           utterance=utterance,
                           utterance_remainder=intent["utterance_remainder"],
                           skill_id=skill_id)

    def _train(self, single_thread=True, timeout=120, force_training=True):
        with self.lock:
            for lang in self.engines:
                self.engines[lang].train(single_thread=single_thread,
                                         timeout=timeout,
                                         force=force_training,
                                         debug=True)
