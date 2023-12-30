package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.ArrayList;


public class a_deck_builder {

	public a_deck provides_Keep_The_Peace() {
		
    	ArrayList<a_card> The_List_Of_Cards_For_Keep_The_Peace = new ArrayList<a_card>();
    	
    	for (int i = 0; i < 25; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_land_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 0, 0, 0, 0),
    				"Plains",
    				"Plains",
    				"Basic Land"
    			)
    		);
    	}
    	// 25 cards
    	
    	for (int i = 0; i < 4; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 0, 0, 0, 1),
    				"Charmed Stray",
    				1,
    				"Common",
    				"Cat",
    				new String[] {
    					"Lifelink",
    					"When Charmed Stray enters the battlefield, put a +1/+1 counter on each other creature you control named Charmed Stray."
    				},
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 29 cards
    	
    	for (int i = 0; i < 4; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new an_instant_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 0, 0, 0, 1),
    				"Tactical Advantage",
    				"Common",
    				new String[] {
    					"Target blocking or blocked creature you control gets +2/+2 until end of turn."
    				},
    				"Instant"
    			)
    		);
    	}
    	// 33 cards
    	
    	for (int i = 0; i < 2; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 1, 0, 0, 1),
    				"Fencing Ace",
    				1,
    				"Common",
    				"Human Soldier",
    				new String[] {
    					"Double strike"
    				},
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 35 cards
    	
    	for (int i = 0; i < 4; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 1, 0, 0, 1),
    				"Hallowed Priest",
    				1,
    				"Uncommon",
    				"Human Cleric",
    				new String[] {
    					"Whenever you gain life, put a +1/+1 counter on Hallowed Priest."
    				},
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 39 cards
    	
    	for (int i = 0; i < 3; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 1, 0, 0, 1),
    				"Impassioned Orator",
    				2,
    				"Common",
    				"Human Cleric",
    				new String[] {
    					"Whenever another creature enters the battlefield under your control, you gain 1 life."
    				},
    				2,
    				"Creature"
    			)
    		);
    	}
    	// 42 cards
    	
    	for (int i = 0; i < 2; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 1, 0, 0, 1),
    				"Moorland Inquisitor",
    				2,
    				"Common",
    				"Human Soldier",
    				new String[] {
    					"[2W]: Moorland Inquisitor gains first strike until end of turn."
    				},
    				2,
    				"Creature"
    			)
    		);
    	}
    	// 44 cards
    	
    	for (int i = 0; i < 3; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new an_aura_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 1, 0, 0, 1),
    				"Pacifism",
    				"Common",
    				"Aura",
    				new String[] {
    					"Enchant creature",
    					"Enchanted creature can't attack or block."
    				},
    				"Enchantment"
    			)
    		);
    	}
    	// 47 cards
    	
    	for (int i = 0; i < 2; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 2, 0, 0, 1),
    				"Angel of Vitality",
    				2,
    				"Uncommon",
    				"Angel",
    				new String[] {
    					"Flying",
    					"If you would gain life, you gain that much life plus 1 instead.",
    					"Angel of Vitality gets +2/+2 as long as you have 25 or more life."
    				},
    				2,
    				"Creature"
    			)
    		);
    	}
    	// 49 cards
    	
    	for (int i = 0; i < 2; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 2, 0, 0, 2),
    				"Leonin Warleader",
    				4,
    				"Rare",
    				"Cat Soldier",
    				new String[] {
    					"Whenever Leonin Warleader attacks, create two 1/1 white Cat creature tokens with lifelink that are tapped and attacking."
    				},
    				4,
    				"Creature"
    			)
    		);
    	}
    	// 51 cards
    	
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new an_aura_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 3, 0, 0, 2),
				"Angelic Reward",
				"Uncommon",
				"Aura",
				new String[] {
					"Enchant creature",
					"Enchanted creature get +3/+3 and has flying."
				},
				"Enchantment"
			)
		);
		// 52 cards
		
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_sorcery_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 4, 0, 0, 1),
				"Bond of Discipline",
				"Uncommon",
				new String[] {
					"Tap all creatures your opponents control. Creatures you control gain lifelink until end of turn."
				},
				"Sorcery"
			)
		);
		// 53 cards
		
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new an_instant_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 4, 0, 0, 1),
				"Confront the Assault",
				"Uncommon",
				new String[] {
					"Cast this spell only if a creature is attacking you.",
					"Create three 1/1 white Spirit creature tokens with flying."
				},
				"Instant"
			)
		);
		// 54 cards
		
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 3, 0, 0, 2),
				"Serra Angel",
				4,
				"Uncommon",
				"Angel",
				new String[] {
					"Flying, Vigilance"
				},
				4,
				"Creature"
			)
		);
		// 55 cards
		
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 3, 0, 0, 2),
				"Spiritual Guardian",
				3,
				"Common",
				"Spirit",
				new String[] {
					"When Spiritual Guardian enters the battlefield, you gain 4 life."
				},
				4,
				"Creature"
			)
		);
		// 56 cards
		
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 4, 0, 0, 2),
				"Angelic Guardian",
				5,
				"Rare",
				"Angel",
				new String[] {
					"Flying",
					"Whenever one or more creatures you control attack, they gain indestructible until end of turn."
				},
				5,
				"Creature"
			)
		);
		// 57 cards
		
    	for (int i = 0; i < 2; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 4, 0, 0, 2),
    				"Inspiring Commander",
    				1,
    				"Rare",
    				"Human Soldier",
    				new String[] {
    					"Whenever another creature with power 2 or less enters the battlefield under your control, you gain 1 life and draw a card."
    				},
    				4,
    				"Creature"
    			)
    		);
    	}
    	// 59 cards
    	
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 5, 0, 0, 2),
				"Goring Ceratops",
				3,
				"Rare",
				"Dinosaur",
				new String[] {
					"Double strike",
					"Whenever Goring Ceratops attacks, other creatures you control gain double strike until end of turn."
				},
				3,
				"Creature"
			)
		);
    	// 60 cards
    	
    	return new a_deck(The_List_Of_Cards_For_Keep_The_Peace, "Keep the Peace");
	}
	
	
	public a_deck provides_Large_And_In_Charge() {
		
    	ArrayList<a_card> The_List_Of_Cards_For_Large_And_In_Charge = new ArrayList<a_card>();
    	
    	for (int i = 0; i < 25; i++) {
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new a_land_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 0, 0, 0, 0),
    				"Forest",
    				"Forest",
    				"Basic Land"
    			)
    		);
    	}
    	// 25 cards
    	
    	for (int i = 0; i < 3; i++) {
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 0, 1, 0, 0),
    				"Jungle Delver",
    				1,
    				"Common",
    				"Merfolk Warrior",
    				new String[] {
    					"[3G]: Put a +1/+1 counter on Jungle Delver."
    				},
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 28 cards
    	
    	for (int i = 0; i < 2; i++) {
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new an_instant_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 0, 1, 0, 0),
    				"Stony Strength",
    				"Common",
    				new String[] {
    					"Put a +1/+1 counter on target creature you control. Untap that creature."
    				},
    				"Instant"
    			)
    		);
    	}
    	// 30 cards
    	
    	for (int i = 0; i < 4; i++) {
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 1, 1, 0, 0),
    				"Ilysian Caryatid",
    				1,
    				"Common",
    				"Plant",
    				new String[] {
    					"[T]: Add one mana of any color. If you control a creature with power 4 or greater, add two mana of any one color instead."
    				},
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 34 cards
		
    	for (int i = 0; i < 3; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_sorcery_card(
					"Arena Base Set",
					new a_mana_pool(0, 0, 1, 1, 0, 0),
					"Rabid Bite",
					"Common",
					new String[] {
						"Target creature you control deals damage equal to its power to target creature you don't control."
					},
					"Sorcery"
				)
			);
    	}
		// 37 cards
		
    	for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_pool(0, 0, 1, 1, 0, 0),
					"Woodland Mystic",
					1,
					"Common",
					"Elf Druid",
					new String[] {
						"[T]: Add [F]."
					},
					1,
					"Creature"
				)
			);
    	}
		// 39 cards
    	
    	for (int i = 0; i < 2; i++) {
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new an_enchantment_card(
    				"Arena Base Set",
    				new a_mana_pool(0, 0, 2, 1, 0, 0),
    				"Colossal Majesty",
    				"Uncommon",
    				new String[] {
    					"At the beginning of your upkeep, if you control a creature with power 4 or greater, draw a card."
    				},
    				"Enchantment"
    			)
    		);
    	}
    	// 41 cards
    	
    	for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_pool(0, 0, 2, 1, 0, 0),
					"Wildwood Patrol",
					4,
					"Common",
					"Centaur Scout",
					new String[] {
						"Trample"
					},
					2,
					"Creature"
				)
			);
    	}
		// 43 cards
    	
    	for (int i = 0; i < 4; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_pool(0, 0, 3, 1, 0, 0),
					"Baloth Packhunter",
					3,
					"Common",
					"Beast",
					new String[] {
						"Trample",
						"When Baloth Packhunter enters the battlefield, put two +1/+1 counters on each other creature you control named Baloth Packhunter."
					},
					3,
					"Creature"
				)
			);
    	}
		// 47 cards
    	
		The_List_Of_Cards_For_Large_And_In_Charge.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 3, 1, 0, 0),
				"Prized Unicorn",
				2,
				"Uncommon",
				"Unicorn",
				new String[] {
					"All creatures able to block Prized Unicorn do so."
				},
				2,
				"Creature"
			)
		);
		// 48 cards
		
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_pool(0, 0, 2, 2, 0, 0),
					"Rumbling Baloth",
					4,
					"Common",
					"Beast",
					new String[] {},
					4,
					"Creature"
				)
			);
		}
		// 50 cards
		
		The_List_Of_Cards_For_Large_And_In_Charge.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 3, 1, 0, 0),
				"World Shaper",
				3,
				"Rare",
				"Merfolk Shaman",
				new String[] {
					"Whenever World Shaper attacks, you may mill three cards.",
					"When World Shaper dies, return all land cards from your graveyard to the battlefield tapped."
				},
				3,
				"Creature"
			)
		);
		// 51 cards
		
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_pool(0, 0, 0, 5, 0, 0),
					"Gigantosaurus",
					10,
					"Rare",
					"Dinosaur",
					new String[] {},
					10,
					"Creature"
				)
			);
		}
		// 53 cards
		
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_pool(0, 0, 3, 2, 0, 0),
					"Sentinel Spider",
					4,
					"Uncommon",
					"Spider",
					new String[] {"Vigilance, Reach"},
					4,
					"Creature"
				)
			);
		}
		// 55 cards
		
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_pool(0, 0, 5, 1, 0, 0),
					"Affectionate Indrik",
					4,
					"Uncommon",
					"Beast",
					new String[] {"When Affectionate Indrik enters the battlefield, you may have it fight target creature you don't control."},
					4,
					"Creature"
				)
			);
		}
		// 57 cards
    	
		The_List_Of_Cards_For_Large_And_In_Charge.add(
			new an_aura_card(
				"Arena Base Set",
				new a_mana_pool(0, 0, 4, 2, 0, 0),
				"Epic Proportions",
				"Rare",
				"Aura",
				new String[] {
					"Flash",
					"Enchant creature",
					"Enchanted creature gets +5/+5 and has trample."
				},
				"Enchantment"
			)
		);
		// 58 cards
		
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_pool(0, 0, 5, 2, 0, 0),
					"Rampaging Brontodon",
					7,
					"Rare",
					"Dinosaur",
					new String[] {
						"Trample",
						"Whenever Rampaging Brontodon attacks, it gets +1/+1 until end of turn for each land you control."},
					7,
					"Creature"
				)
			);
		}
		// 60 cards
		
    	return new a_deck(The_List_Of_Cards_For_Large_And_In_Charge, "Large_and_in_Charge");
	}
	
}
