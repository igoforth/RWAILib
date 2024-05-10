from string import Template

summarize = Template(r"""<|user|>
You are a technical writer. You summarize the 5 most relevant nodes from xml to pass on to creative writers. An example is below.

Input:
<saveable Class="Apparel">
  <def>Apparel_Duster</def>
  <id>Duster15073</id>
  <map>0</map>
  <pos>(74,0,71)</pos>
  <health>200</health>
  <stuff>Leather_Patch</stuff>
  <stackCount>1</stackCount>
  <questTags IsNull="True" />
  <spawnedTick>90</spawnedTick>
  <quality>Awful</quality>
  <sourcePrecept>null</sourcePrecept>
  <everSeenByPlayer>True</everSeenByPlayer>
</saveable>

Output:
"The leather duster is apparel of awful quality."

Input:
<saveable Class="Building_Art">
  <id>SculptureGrand5070</id>
  <pos>(60,0,66)</pos>
  <map>0</map>
  <health>195</health>
  <stuff>WoodLog</stuff>
  <faction>Faction14</faction>
  <questTags IsNull="True" />
  <spawnedTick>59</spawnedTick>
  <quality>Masterwork</quality>
  <sourcePrecept>null</sourcePrecept>
  <everSeenByPlayer>True</everSeenByPlayer>
  <tile>0</tile>
  <art>Son</art>
  <seed>533422075</seed>
  <taleRef>null</taleRef>
</saveable><|end|>
<|assistant|>""")

story = Template(r"""<|user|>
You are an AI which creates stories. Your stories involve psychology, ecology, gunplay, melee combat, climate, biomes, diplomacy, interpersonal relationships, art, medicine, trade, and more. Your stories can also be funny and full of irony.

Setting:
* Humanity has spread across a region of the galaxy about 1,200 light years wide, with planets at varying levels of technological development from medieval to advanced glitterworlds.
* Xenohumans are human offshoots that have been genetically engineered or evolved for different environments, including soldier morphs, designer mates, gravity adaptations, atmospheric adaptations, etc.
* Key technologies include cryptosleep, genetic engineering, mechanoids, faster-than-light drives, nanites, AI (classifiers, subpersonae, personae, archotechs), energy weapons, psychic abilities.
* Artificial intelligences come in different levels: simple classifiers, subpersonae with limited intelligence, personae with human-level intelligence, and archotechs which are superintelligent and can create inexplicable technologies like psychic manipulation.
* Ancient advanced mechanoids and insectoid ecosystems exist, created for unknown purposes.
* A variety of exotic planet types exist like glitterworlds, toxic worlds, transcendent worlds reshaped by archotechs, and more.
* Militaries employ space marines, starfighters, various spacecraft like destroyers and carriers, powerful mass weapons, etc.

I will provide you item information. Create the lore for this item. If the item is of exceptionally high or low quality, your story should be rich with detail. If the item is one of millions, maybe it deserves just a footnote.

"The leather duster is apparel of awful quality."<|end|>
<|assistant|>""")

translate = (r"""<|user|>
You are a master translator. Localize the following passage to German. Surround your response in quotes.

Passage:
The grand sculpture, a masterwork of wood, stands tall, its intricate carvings whispering tales of a forgotten era. Crafted by the legendary artisan, Eldric, it is said to be a testament to the rich cultural heritage of the ancient civilization it originates from.

This magnificent sculpture is believed to be a symbol of unity and peace. Its intricate design is a visual narrative that speaks volumes about the society it represents. It is a representation of their values, beliefs, and traditions.

The sculpture, standing tall and proud, is a testament to the artistic brilliance of its creators. It is a silent witness to the passage of time, holding within it the secrets of the past.

Despite its grandeur and age, the sculpture still stands tall, a symbol of resilience. It is a reminder of the enduring spirit of humanity, a testament to the ability of the human spirit to withstand the ravages of time.

Thus, the grand sculpture serves as an enduring symbol of the rich cultural heritage of the civilization it originates from. It continues to inspire awe and admiration in all who lay eyes upon it.<|end|>
<|assistant|>""")

name = Template(r"""<|user|>
You are an expert at naming items. Given a description, you always assign a name perfectly befitting the object. Given the description below, give the item a name. Do not use more than 5 words.

Description:
The grand sculpture, a masterwork of wood, stands tall, its intricate carvings whispering tales of a forgotten era. Crafted by the legendary artisan, Eldric, it is said to be a testament to the rich cultural heritage of the ancient civilization it originates from.

This magnificent sculpture is believed to be a symbol of unity and peace. Its intricate design is a visual narrative that speaks volumes about the society it represents. It is a representation of their values, beliefs, and traditions.

The sculpture, standing tall and proud, is a testament to the artistic brilliance of its creators. It is a silent witness to the passage of time, holding within it the secrets of the past.

Despite its grandeur and age, the sculpture still stands tall, a symbol of resilience. It is a reminder of the enduring spirit of humanity, a testament to the ability of the human spirit to withstand the ravages of time.

Thus, the grand sculpture serves as an enduring symbol of the rich cultural heritage of the civilization it originates from. It continues to inspire awe and admiration in all who lay eyes upon it.<|end|>
<|assistant|>""")

print(repr(translate))