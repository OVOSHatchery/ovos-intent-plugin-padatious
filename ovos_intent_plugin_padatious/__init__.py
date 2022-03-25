from padatious import IntentContainer
from os.path import join, expanduser
from threading import Lock
from ovos_plugin_manager.templates.intents import IntentExtractor
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home


class PadatiousExtractor(IntentExtractor):
    keyword_based = False

    def __init__(self, cache_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_dir = expanduser(self.config.get("data_dir", xdg_data_home()))
        cache_dir = cache_dir or join(data_dir, "padatious")
        self.lock = Lock()
        self.container = IntentContainer(cache_dir)
        self.registered_intents = []

    def detach_intent(self, intent_name):
        if intent_name in self.registered_intents:
            LOG.debug("Detaching padatious intent: " + intent_name)
            with self.lock:
                self.container.remove_intent(intent_name)
            self.registered_intents.remove(intent_name)

    def detach_skill(self, skill_id):
        LOG.debug("Detaching padatious skill: " + str(skill_id))
        remove_list = [i for i in self.registered_intents if skill_id in i]
        for i in remove_list:
            self.detach_intent(i)

    def register_entity(self, entity_name, samples=None, reload_cache=True):
        samples = samples or [entity_name]
        with self.lock:
            self.container.add_entity(entity_name, samples,
                                  reload_cache=reload_cache)

    def register_intent(self, intent_name, samples=None, reload_cache=True):
        samples = samples or [intent_name]
        if intent_name not in self._intent_samples:
            self._intent_samples[intent_name] = samples
        else:
            self._intent_samples[intent_name] += samples
        with self.lock:
            self.container.add_intent(intent_name, samples,
                                  reload_cache=reload_cache)
        self.registered_intents.append(intent_name)

    def register_entity_from_file(self, entity_name, file_name,
                                  reload_cache=True):
        with self.lock:
            self.container.load_entity(entity_name, file_name,
                                       reload_cache=reload_cache)

    def register_intent_from_file(self, intent_name, file_name,
                                  single_thread=True, timeout=120,
                                  reload_cache=True, force_training=True):
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

    def calc_intent(self, utterance, min_conf=None):
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

    def intent_scores(self, utterance):
        utterance = utterance.strip().lower()
        intents = [i.__dict__ for i in self.container.calc_intents(utterance)]
        for idx, intent in enumerate(intents):
            intent["utterance_remainder"] = self._get_remainder(intent, utterance)
            intents[idx]["entities"] = intents[idx].pop("matches")
            intents[idx]["intent_type"] = intents[idx].pop("name")
            intent["intent_engine"] = "padatious"
            intent["utterance"] = intent.pop("sent")
            if isinstance(intents[idx]["utterance"], list):
                intents[idx]["utterance"] = " ".join(intents[idx]["utterance"])
        return intents

    def calc_intents(self, utterance, min_conf=None):
        min_conf = min_conf or self.config.get("padatious_min_conf", 0.35)
        utterance = utterance.strip().lower()
        bucket = {}
        for ut in self.segmenter.segment(utterance):
            intent = self.calc_intent(ut)
            if intent["conf"] < min_conf:
                bucket[ut] = None
            else:
                bucket[ut] = intent
        return bucket

    def calc_intents_list(self, utterance):
        utterance = utterance.strip().lower()
        bucket = {}
        for ut in self.segmenter.segment(utterance):
            bucket[ut] = self.filter_intents(ut)
        return bucket

    def manifest(self):
        # TODO vocab, skill ids, intent_data
        return {
            "intent_names": self.registered_intents
        }

    def _train(self, single_thread=True, timeout=120, force_training=True):
        with self.lock:
            return self.container.train(single_thread=single_thread,
                                        timeout=timeout,
                                        force=force_training,
                                        debug=True)


if __name__ == "__main__":
    from pprint import pprint

    intents = PadatiousExtractor()

    weather = ["weather"]
    hello = ["hey", "hello", "hi", "greetings"]
    name = ["my name is {name}"]
    joke = ["tell me a joke", "i want a joke", "say a joke", "tell joke"]
    lights_on = ["turn on the lights", "lights on", "turn lights on",
                 "turn the lights on"]
    lights_off = ["turn off the lights", "lights off", "turn lights off",
                  "turn the lights off"]
    door_on = ["open the door", "open door", "open the doors"]
    door_off = ["close the door", "close door", "close the doors"]
    music = ["play music", "play some songs", "play heavy metal",
             "play some jazz", "play rock", "play some music"]
    pizza = ["order pizza", "get pizza", "buy pizza"]
    call = ["call {person}", "phone {person}"]
    greet_person = ["say hello to {person}", "tell {person} hello",
                    "tell {person} i said hello"]

    intents.register_intent("weather", weather)
    intents.register_intent("hello", hello)
    intents.register_intent("name", name)
    intents.register_intent("joke", joke)
    intents.register_intent("lights_on", lights_on)
    intents.register_intent("lights_off", lights_off)
    intents.register_intent("door_open", door_on)
    intents.register_intent("door_close", door_off)
    intents.register_intent("play_music", music)
    intents.register_intent("pizza", pizza)
    intents.register_intent("greet_person", greet_person)
    intents.register_intent("call_person", call)

    sentences = [
        "tell me a joke and say hello",
        "turn off the lights, open the door",
        "nice work! get me a beer",
        "Call mom and tell her hello",
        "tell me a joke and the weather",
        "turn on the lights close the door",
        "close the door turn off the lights",
        "tell me a joke and order some pizza and turn on the lights and close the door and play some songs",
        "close the pod bay doors play some music"  # fail
    ]

    print("CALCULATE SINGLE INTENT")
    for sent in sentences:
        print("UTTERANCE:", sent)
        pprint(intents.calc_intent(sent))
        print("_______________________________")

    print("SEGMENT AND CALCULATE BEST INTENTS")
    for sent in sentences:
        print("UTTERANCE:", sent)
        pprint(intents.calc_intents(sent))
        print("_______________________________")

    print("FILTER BY SCORE INTENTS")
    for sent in sentences:
        print("UTTERANCE:", sent)
        pprint(intents.filter_intents(sent))
        print("_______________________________")

    print("SEGMENT AND FILTER")
    for sent in sentences:
        print("UTTERANCE:", sent)
        pprint(intents.calc_intents_list(sent))
        print("_______________________________")
