from threading import Lock

from ovos_config import Configuration
from ovos_plugin_manager.templates.pipeline import IntentPipelinePlugin, IntentMatch
from ovos_utils import classproperty
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home
from padatious import IntentContainer
from ovos_config.meta import get_xdg_base


def _munge(name, skill_id):
    return f"{name}:{skill_id}"


def _unmunge(munged):
    return munged.split(":", 2)


class PadatiousPipelinePlugin(IntentPipelinePlugin):

    def __init__(self, bus, config=None):
        config = config or Configuration().get("padatious", {})  # deprecated
        super().__init__(bus, config)
        self.cache_dir = self.config.get('intent_cache') or \
                         f"{xdg_data_home()}/{get_xdg_base()}/padatious_cache"
        self.lock = Lock()
        self.engines = {}  # lang: IntentContainer

    # plugin api
    @classproperty
    def matcher_id(self):
        return "padatious"

    def match(self, utterances, lang, message=None):
        return self.calc_intent(utterances, lang=lang)

    def train(self, single_thread=True, timeout=120, force_training=True):
        with self.lock:
            try:
                for lang in self.engines:
                    self.engines[lang].train(single_thread=single_thread,
                                             timeout=timeout,
                                             force=force_training,
                                             debug=True)
            except Exception as e:
                LOG.exception(f"failed to train {lang}")
                return False

        return True

    # implementation
    def _get_engine(self, lang=None):
        lang = lang or self.lang
        if lang not in self.engines:
            self.engines[lang] = IntentContainer(self.cache_dir)
        return self.engines[lang]

    def detach_intent(self, skill_id, intent_name):
        LOG.debug("Detaching padatious intent: " + intent_name)
        with self.lock:
            munged = _munge(intent_name, skill_id)
            for lang in self.engines:
                self.engines[lang].remove_intent(munged)
        super().detach_intent(intent_name)

    def detach_entity(self, skill_id, entity_name):
        LOG.debug("Detaching padatious entity: " + entity_name)
        name = _munge(entity_name, skill_id)
        with self.lock:
            for lang in self.engines:
                self.engines[lang].remove_entity(name)
        super().detach_entity(skill_id, entity_name)

    def detach_skill(self, skill_id):
        LOG.debug("Detaching padatious skill: " + skill_id)
        with self.lock:
            for lang in self.engines:
                for entity in (e for e in self.registered_entities if e.skill_id == skill_id):
                    munged = _munge(entity.name, skill_id)
                    self.engines[lang].remove_entity(munged)
                for intent in (e for e in self.registered_intents if e.skill_id == skill_id):
                    munged = _munge(intent.name, skill_id)
                    self.engines[lang].remove_intent(munged)
        super().detach_skill(skill_id)

    def register_entity(self, skill_id, entity_name, samples=None, lang=None, reload_cache=True):
        lang = lang or self.lang
        super().register_entity(skill_id, entity_name, samples, lang)
        container = self._get_engine(lang)
        samples = samples or [entity_name]
        with self.lock:
            container.add_entity(entity_name, samples, reload_cache=reload_cache)

    def register_intent(self, skill_id, intent_name, samples=None, lang=None, reload_cache=True,
                        single_thread=True, timeout=120, force_training=False):
        lang = lang or self.lang
        super().register_intent(skill_id, intent_name, samples, lang)
        container = self._get_engine(lang)
        samples = samples or [intent_name]
        intent_name = _munge(intent_name, skill_id)
        with self.lock:
            container.add_intent(intent_name, samples,
                                 reload_cache=reload_cache)

        success = self.train(single_thread=single_thread,
                             timeout=timeout,
                             force_training=force_training)
        if success:
            LOG.debug(intent_name + " trained successfully")
        else:
            LOG.error(intent_name + " FAILED TO TRAIN")

    def calc_intent(self, utterance, min_conf=0.0, lang=None):
        lang = lang or self.lang
        container = self._get_engine(lang)
        min_conf = min_conf or self.config.get("padatious_min_conf", 0.35)
        utterance = utterance.strip().lower()
        with self.lock:
            intent = container.calc_intent(utterance).__dict__
        if intent["conf"] < min_conf:
            return None
        intent_type, skill_id = _unmunge(intent["intent_type"])
        return IntentMatch(intent_service=self.matcher_id,
                           intent_type=intent_type,
                           intent_data=intent.pop("matches"),
                           confidence=intent["conf"],
                           utterance=utterance,
                           skill_id=skill_id)
