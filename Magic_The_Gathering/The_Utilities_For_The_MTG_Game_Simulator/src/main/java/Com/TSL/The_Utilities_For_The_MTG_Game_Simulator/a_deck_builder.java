package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_deck_builder {

	public a_deck builds_Keep_The_Peace() {
		
    	ArrayList<a_card> The_List_Of_Cards_For_Keep_The_Peace = new ArrayList<a_card>();
    	
    	for (int i = 0; i < 100; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_land_card(
    				"Arena Base Set",
    				"Plains",
    				"Plains",
    				new ArrayList<String>(),
    				"Basic Land"
    			)
    		);
    	}
    	// 25 cards
    	
    	for (int i = 0; i < 100; i++) {
    		ArrayList<String> The_Text = new ArrayList<>();
    		The_Text.add("Lifelink");
    		The_Text.add("When Charmed Stray enters the battlefield, put a +1/+1 counter on each other creature you control named Charmed Stray.");
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 0, 0, 0, 1),
    				"Charmed Stray",
    				1,
    				"Common",
    				"Cat",
    				The_Text,
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 29 cards
    	
    	for (int i = 0; i < 4; i++) {
    		ArrayList<String> The_Text = new ArrayList<>();
    		The_Text.add("Target blocking or blocked creature you control gets +2/+2 until end of turn.");
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new an_instant_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 0, 0, 0, 1),
    				"Tactical Advantage",
    				"Common",
    				The_Text,
    				"Instant"
    			)
    		);
    	}
    	// 33 cards
    	
    	for (int i = 0; i < 2; i++) {
    		ArrayList<String> The_Text = new ArrayList<>();
    		The_Text.add("Double strike");
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 1, 0, 0, 1),
    				"Fencing Ace",
    				1,
    				"Common",
    				"Human Soldier",
    				The_Text,
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 35 cards
    	
    	for (int i = 0; i < 4; i++) {
    		ArrayList<String> The_Text = new ArrayList<>();
    		The_Text.add("Whenever you gain life, put a +1/+1 counter on Hallowed Priest.");
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 1, 0, 0, 1),
    				"Hallowed Priest",
    				1,
    				"Uncommon",
    				"Human Cleric",
    				The_Text,
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 39 cards
    	
    	for (int i = 0; i < 3; i++) {
    		ArrayList<String> The_Text = new ArrayList<>();
    		The_Text.add("Whenever another creature enters the battlefield under your control, you gain 1 life.");
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 1, 0, 0, 1),
    				"Impassioned Orator",
    				2,
    				"Common",
    				"Human Cleric",
    				The_Text,
    				2,
    				"Creature"
    			)
    		);
    	}
    	// 42 cards
    	
    	for (int i = 0; i < 2; i++) {
    		ArrayList<String> The_Text = new ArrayList<>();
    		The_Text.add("[2W]: Moorland Inquisitor gains first strike until end of turn.");
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 1, 0, 0, 1),
    				"Moorland Inquisitor",
    				2,
    				"Common",
    				"Human Soldier",
    				The_Text,
    				2,
    				"Creature"
    			)
    		);
    	}
    	// 44 cards
    	
    	for (int i = 0; i < 3; i++) {
    		ArrayList<String> The_Text = new ArrayList<>();
    		The_Text.add("Enchant creature");
    		The_Text.add("Enchanted creature can't attack or block.");
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new an_aura_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 1, 0, 0, 1),
    				"Pacifism",
    				"Common",
    				"Aura",
    				The_Text,
    				"Enchantment"
    			)
    		);
    	}
    	// 47 cards
    	
    	for (int i = 0; i < 2; i++) {
    		ArrayList<String> The_Text = new ArrayList<>();
    		The_Text.add("Flying");
    		The_Text.add("If you would gain life, you gain that much life plus 1 instead.");
    		The_Text.add("Angel of Vitality gets +2/+2 as long as you have 25 or more life.");
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 2, 0, 0, 1),
    				"Angel of Vitality",
    				2,
    				"Uncommon",
    				"Angel",
    				The_Text,
    				2,
    				"Creature"
    			)
    		);
    	}
    	// 49 cards
    	
    	for (int i = 0; i < 2; i++) {
    		ArrayList<String> The_Text = new ArrayList<>();
    		The_Text.add("Whenever Leonin Warleader attacks, create two 1/1 white Cat creature tokens with lifelink that are tapped and attacking.");
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 2, 0, 0, 2),
    				"Leonin Warleader",
    				4,
    				"Rare",
    				"Cat Soldier",
    				The_Text,
    				4,
    				"Creature"
    			)
    		);
    	}
    	// 51 cards
    	
		ArrayList<String> The_Text = new ArrayList<>();
		The_Text.add("Enchant creature");
		The_Text.add("Enchanted creature get +3/+3 and has flying.");
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new an_aura_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 3, 0, 0, 2),
				"Angelic Reward",
				"Uncommon",
				"Aura",
				The_Text,
				"Enchantment"
			)
		);
		// 52 cards
		
    	The_Text = new ArrayList<String>();
    	The_Text.add("Tap all creatures your opponents control. Creatures you control gain lifelink until end of turn.");
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_sorcery_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 4, 0, 0, 1),
				"Bond of Discipline",
				"Uncommon",
				The_Text,
				"Sorcery"
			)
		);
		// 53 cards
		
    	The_Text = new ArrayList<>();
    	The_Text.add("Cast this spell only if a creature is attacking you.");
    	The_Text.add("Create three 1/1 white Spirit creature tokens with flying.");
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new an_instant_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 4, 0, 0, 1),
				"Confront the Assault",
				"Uncommon",
				The_Text,
				"Instant"
			)
		);
		// 54 cards
		
    	The_Text = new ArrayList<>();
    	The_Text.add("Flying, Vigilance");
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 3, 0, 0, 2),
				"Serra Angel",
				4,
				"Uncommon",
				"Angel",
				The_Text,
				4,
				"Creature"
			)
		);
		// 55 cards
		
    	The_Text = new ArrayList<>();
    	The_Text.add("When Spiritual Guardian enters the battlefield, you gain 4 life.");
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 3, 0, 0, 2),
				"Spiritual Guardian",
				3,
				"Common",
				"Spirit",
				The_Text,
				4,
				"Creature"
			)
		);
		// 56 cards
		
    	The_Text = new ArrayList<>();
    	The_Text.add("Flying");
    	The_Text.add("Whenever one or more creatures you control attack, they gain indestructible until end of turn.");
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 4, 0, 0, 2),
				"Angelic Guardian",
				5,
				"Rare",
				"Angel",
				The_Text,
				5,
				"Creature"
			)
		);
		// 57 cards
		
    	The_Text = new ArrayList<>();
    	The_Text.add("Whenever another creature with power 2 or less enters the battlefield under your control, you gain 1 life and draw a card.");
    	for (int i = 0; i < 2; i++) {
    		The_List_Of_Cards_For_Keep_The_Peace.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 4, 0, 0, 2),
    				"Inspiring Commander",
    				1,
    				"Rare",
    				"Human Soldier",
    				The_Text,
    				4,
    				"Creature"
    			)
    		);
    	}
    	// 59 cards
    	
    	The_Text = new ArrayList<>();
    	The_Text.add("Double strike");
    	The_Text.add("Whenever Goring Ceratops attacks, other creatures you control gain double strike until end of turn.");
    	The_List_Of_Cards_For_Keep_The_Peace.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 5, 0, 0, 2),
				"Goring Ceratops",
				3,
				"Rare",
				"Dinosaur",
				The_Text,
				3,
				"Creature"
			)
		);
    	// 60 cards
    	
    	return new a_deck(The_List_Of_Cards_For_Keep_The_Peace, "Keep the Peace");
	}
	
	
	public a_deck builds_Large_And_In_Charge() {
		
    	ArrayList<a_card> The_List_Of_Cards_For_Large_And_In_Charge = new ArrayList<a_card>();
    	
    	ArrayList<String> The_Text;
    	for (int i = 0; i < 100; i++) {
        	The_Text = new ArrayList<>();
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new a_land_card(
    				"Arena Base Set",
    				"Forest",
    				"Forest",
    				The_Text,
    				"Basic Land"
    			)
    		);
    	}
    	// 25 cards
    	
    	for (int i = 0; i < 100; i++) {
    		The_Text = new ArrayList<>();
    		The_Text.add("[3G]: Put a +1/+1 counter on Jungle Delver.");
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 0, 1, 0, 0),
    				"Jungle Delver",
    				1,
    				"Common",
    				"Merfolk Warrior",
    				The_Text,
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 28 cards
    	
    	for (int i = 0; i < 2; i++) {
    		The_Text = new ArrayList<>();
    		The_Text.add("Put a +1/+1 counter on target creature you control. Untap that creature.");
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new an_instant_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 0, 1, 0, 0),
    				"Stony Strength",
    				"Common",
    				The_Text,
    				"Instant"
    			)
    		);
    	}
    	// 30 cards
    	
    	for (int i = 0; i < 4; i++) {
    		The_Text = new ArrayList<>();
    		The_Text.add("[T]: Add one mana of any color. If you control a creature with power 4 or greater, add two mana of any one color instead.");
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new a_creature_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 1, 1, 0, 0),
    				"Ilysian Caryatid",
    				1,
    				"Common",
    				"Plant",
    				The_Text,
    				1,
    				"Creature"
    			)
    		);
    	}
    	// 34 cards
		
    	for (int i = 0; i < 3; i++) {
    		The_Text = new ArrayList<>();
    		The_Text.add("Target creature you control deals damage equal to its power to target creature you don't control.");
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_sorcery_card(
					"Arena Base Set",
					new a_mana_cost(0, 0, 1, 1, 0, 0),
					"Rabid Bite",
					"Common",
					The_Text,
					"Sorcery"
				)
			);
    	}
		// 37 cards
		
    	for (int i = 0; i < 2; i++) {
    		The_Text = new ArrayList<>();
    		The_Text.add("[T]: Add [F].");
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_cost(0, 0, 1, 1, 0, 0),
					"Woodland Mystic",
					1,
					"Common",
					"Elf Druid",
					The_Text,
					1,
					"Creature"
				)
			);
    	}
		// 39 cards
    	
		The_Text = new ArrayList<>();
		The_Text.add("At the beginning of your upkeep, if you control a creature with power 4 or greater, draw a card.");
    	for (int i = 0; i < 2; i++) {
    		The_List_Of_Cards_For_Large_And_In_Charge.add(
    			new an_enchantment_card(
    				"Arena Base Set",
    				new a_mana_cost(0, 0, 2, 1, 0, 0),
    				"Colossal Majesty",
    				"Uncommon",
    				The_Text,
    				"Enchantment"
    			)
    		);
    	}
    	// 41 cards
    	
    	for (int i = 0; i < 2; i++) {
    		The_Text = new ArrayList<>();
    		The_Text.add("Trample");
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_cost(0, 0, 2, 1, 0, 0),
					"Wildwood Patrol",
					4,
					"Common",
					"Centaur Scout",
					The_Text,
					2,
					"Creature"
				)
			);
    	}
		// 43 cards
    	
    	for (int i = 0; i < 4; i++) {
    		The_Text = new ArrayList<>();
    		The_Text.add("Trample");
    		The_Text.add("When Baloth Packhunter enters the battlefield, put two +1/+1 counters on each other creature you control named Baloth Packhunter.");
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_cost(0, 0, 3, 1, 0, 0),
					"Baloth Packhunter",
					3,
					"Common",
					"Beast",
					The_Text,
					3,
					"Creature"
				)
			);
    	}
		// 47 cards
    	
		The_Text = new ArrayList<>();
		The_Text.add("All creatures able to block Prized Unicorn do so.");
		The_List_Of_Cards_For_Large_And_In_Charge.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 3, 1, 0, 0),
				"Prized Unicorn",
				2,
				"Uncommon",
				"Unicorn",
				The_Text,
				2,
				"Creature"
			)
		);
		// 48 cards
		
		The_Text = new ArrayList<>();
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_cost(0, 0, 2, 2, 0, 0),
					"Rumbling Baloth",
					4,
					"Common",
					"Beast",
					The_Text,
					4,
					"Creature"
				)
			);
		}
		// 50 cards
		
		The_Text = new ArrayList<>();
		The_Text.add("Whenever World Shaper attacks, you may mill three cards.");
		The_Text.add("When World Shaper dies, return all land cards from your graveyard to the battlefield tapped.");
		The_List_Of_Cards_For_Large_And_In_Charge.add(
			new a_creature_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 3, 1, 0, 0),
				"World Shaper",
				3,
				"Rare",
				"Merfolk Shaman",
				The_Text,
				3,
				"Creature"
			)
		);
		// 51 cards
		
		The_Text = new ArrayList<>();
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_cost(0, 0, 0, 5, 0, 0),
					"Gigantosaurus",
					10,
					"Rare",
					"Dinosaur",
					The_Text,
					10,
					"Creature"
				)
			);
		}
		// 53 cards
		
		The_Text = new ArrayList<>();
		The_Text.add("Vigilance, Reach");
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_cost(0, 0, 3, 2, 0, 0),
					"Sentinel Spider",
					4,
					"Uncommon",
					"Spider",
					The_Text,
					4,
					"Creature"
				)
			);
		}
		// 55 cards
		
		The_Text = new ArrayList<>();
		The_Text.add("When Affectionate Indrik enters the battlefield, you may have it fight target creature you don't control.");
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_cost(0, 0, 5, 1, 0, 0),
					"Affectionate Indrik",
					4,
					"Uncommon",
					"Beast",
					The_Text,
					4,
					"Creature"
				)
			);
		}
		// 57 cards
    	
		The_Text = new ArrayList<>();
		The_Text.add("Flash");
		The_Text.add("Enchant creature");
		The_Text.add("Enchanted creature gets +5/+5 and has trample.");
		The_List_Of_Cards_For_Large_And_In_Charge.add(
			new an_aura_card(
				"Arena Base Set",
				new a_mana_cost(0, 0, 4, 2, 0, 0),
				"Epic Proportions",
				"Rare",
				"Aura",
				The_Text,
				"Enchantment"
			)
		);
		// 58 cards
		
		The_Text = new ArrayList<>();
		The_Text.add("Trample");
		The_Text.add("Whenever Rampaging Brontodon attacks, it gets +1/+1 until end of turn for each land you control.");
		for (int i = 0; i < 2; i++) {
			The_List_Of_Cards_For_Large_And_In_Charge.add(
				new a_creature_card(
					"Arena Base Set",
					new a_mana_cost(0, 0, 5, 2, 0, 0),
					"Rampaging Brontodon",
					7,
					"Rare",
					"Dinosaur",
					The_Text,
					7,
					"Creature"
				)
			);
		}
		// 60 cards
		
    	return new a_deck(The_List_Of_Cards_For_Large_And_In_Charge, "Large_and_in_Charge");
	}
	
}
