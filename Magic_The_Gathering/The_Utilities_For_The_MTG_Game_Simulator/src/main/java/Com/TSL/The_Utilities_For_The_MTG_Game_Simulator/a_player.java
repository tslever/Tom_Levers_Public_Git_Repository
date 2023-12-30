package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.ArrayList;
import org.apache.commons.math3.random.RandomDataGenerator;


public class a_player
{
	
	/** Rule 103.4:
	 * Each player draws a number of cards equal to their starting hand size, which is normally seven.
	 */
	
	private static int STARTING_HAND_SIZE = 7;
	
	private a_deck Deck;
	private an_exile Exile;
	private a_graveyard Graveyard;
	private a_hand Hand;
	private boolean Has_Priority;
	private int Index_Of_The_Present_Turn;
	private int Life;
	private ArrayList<a_permanent> List_Of_Permanents_That_Should_Be_Untapped;
	private a_mana_pool Mana_Pool;
	private String Name;
	private a_part_of_the_battlefield Part_Of_The_Battlefield;
	private a_player Other_Player;
	private RandomDataGenerator Random_Data_Generator;
	private a_stack Stack;
	private boolean Was_Starting_Player;
	
	
	/** Rule 103.3:
	 * Each player begins the game with a starting life total of 20.
	 */
	
	public a_player(a_deck The_Deck_To_Use, String The_Name_To_Use, a_stack The_Stack_To_Use)
	{
		this.Deck = The_Deck_To_Use;
		this.Exile = new an_exile();
		this.Graveyard = new a_graveyard();
		this.Hand = new a_hand();
		this.Has_Priority = false;
		this.Index_Of_The_Present_Turn = 0;
		this.Life = 20;
		this.List_Of_Permanents_That_Should_Be_Untapped = new ArrayList<a_permanent>();
		this.Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
		this.Name = The_Name_To_Use;
		this.Part_Of_The_Battlefield = new a_part_of_the_battlefield();
		this.Random_Data_Generator = new RandomDataGenerator();
		this.Stack = The_Stack_To_Use;
		this.Was_Starting_Player = false;
	}
	
	
	public void becomes_the_starting_player() {
		this.Was_Starting_Player = true;
	}
	
	
	public a_nonland_card chooses_a_card_to_use_to_cast_a_spell_from(ArrayList<a_nonland_card> The_List_Of_Nonland_Cards_That_May_Be_Used_To_Cast_Spells) {
		int The_Index_Of_The_Nonland_Card_To_Use = this.Random_Data_Generator.nextInt(0, The_List_Of_Nonland_Cards_That_May_Be_Used_To_Cast_Spells.size() - 1);
		a_nonland_card The_Nonland_Card_To_Use = The_List_Of_Nonland_Cards_That_May_Be_Used_To_Cast_Spells.remove(The_Index_Of_The_Nonland_Card_To_Use);
		return The_Nonland_Card_To_Use;
	}
	
	
	public ArrayList<a_configuration_of_a_permanent_to_contribute_mana> chooses_a_sufficient_combination_of_configurations_of_a_permanent_to_contribute_mana_from(ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>> The_List_Of_Sufficient_Combinations_To_Use) {
		int The_Index_Of_The_Sufficient_Combination = this.Random_Data_Generator.nextInt(0, The_List_Of_Sufficient_Combinations_To_Use.size() - 1);
		ArrayList<a_configuration_of_a_permanent_to_contribute_mana> The_Sufficient_Combination = The_List_Of_Sufficient_Combinations_To_Use.get(The_Index_Of_The_Sufficient_Combination);
		return The_Sufficient_Combination;
	}
	
	
	public void completes_her_beginning_phase() {
		
		System.out.println(this.Name + " is completing their beginning phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 501.1: The beginning phase consists of three steps, in this order: untap, upkeep, and draw.
		this.completes_her_untap_step();
		
		this.completes_her_upkeep_step();
		
		// Rule 103.7a: In a two-player game, the player who plays first skips the draw step (see rule 504, "Draw Step") of their first turn.
		if ((this.Was_Starting_Player) && (this.Index_Of_The_Present_Turn == 0)) {
			System.out.println("Because " + this.Name + " is the starting player and this is the first turn, the draw step is skipped.");
		}
		else {
			this.completes_her_draw_step();
		}

		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
	}
	
	
	public void completes_her_combat_phase() {
		
		System.out.println(this.Name + " is completing their combat phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire... Effects that last "until end of combat" expire at the end of the combat phase.
	}
	
	
    public void completes_her_draw_step() {
		
		System.out.println("    " + this.Name + " is completing their draw step.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 504.1: First, the active player draws a card. This turn-based action doesn't use the stack.
		this.draws();
		
		// Rule 504.2: Second, the active player gets priority. (See rule 117, "Timing and Priority.")
		this.Has_Priority = true;
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
		
		this.Has_Priority = false;
	}
	
	
	public void completes_her_end_phase() {
		
		System.out.println(this.Name + " is completing their end phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire... Effects that last "until end of combat" expire at the end of the combat phase.
		
	}
	
	
	public a_mana_pool acquires_mana_for(a_nonland_card The_Nonland_Card_For_Which_To_Acquire_Mana) {
		a_mana_pool The_Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
		ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>> The_List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana = The_Nonland_Card_For_Which_To_Acquire_Mana.provides_its_list_of_sufficient_combinations_of_configurations_of_a_permanent_to_contribute_mana();
		ArrayList<a_configuration_of_a_permanent_to_contribute_mana> The_Sufficient_Combination_Of_Configurations_Of_A_Permanent_To_Contribute_Mana = this.chooses_a_sufficient_combination_of_configurations_of_a_permanent_to_contribute_mana_from(The_List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana);
		for (a_configuration_of_a_permanent_to_contribute_mana The_Configuration_Of_A_Permanent_To_Contribute_Mana : The_Sufficient_Combination_Of_Configurations_Of_A_Permanent_To_Contribute_Mana) {
			a_permanent The_Permanent = The_Configuration_Of_A_Permanent_To_Contribute_Mana.provides_its_permanent();
			int The_Option_For_Contributing_Mana = The_Configuration_Of_A_Permanent_To_Contribute_Mana.provides_its_option_for_contributing_mana();
			The_Mana_Pool.increases_by(The_Permanent.provides_a_mana_pool_for(The_Option_For_Contributing_Mana));
		}
		return The_Mana_Pool;
	}
	
	
	public void completes_her_precombat_main_phase() throws Exception {
		
		System.out.println(this.Name + " is completing their precombat main phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 505.4: Second, if the active player controls one or more Saga enchantments and it's the active player's precombat main phase, the active player puts a lore counter on each Saga they control. (See rule 714, "Saga Cards.") This turn-based action doesn't use the stack.

		// Rule 505.5: Third, the active player gets priority. (See rule 117, "Timing and Priority.")
		this.Has_Priority = true;
		
		// Rule 505.5b: During either main phase, the active player may play one land card from their hand if the stack is empty, if the player has priority, and if they haven't played a land this turn (unless an effect states the player may play additional lands). This action doesn't use the stack. Neither the land nor the action of playing the land is a spell or ability, so it can't be countered, and players can't respond to it with instants or activated abilities. (See rule 305, "Lands.")
		this.plays_a_land();
		
		// Rule 505.5a: [A] main phase is the only phase in which a player can normally cast artifact, creature, enchantment, planeswalker, and sorcery spells. The active player may cast these spells.
		// Rule 601.2: To cast a spell is to [use a card to create a spell], put [the spell] on the stack, and pay its mana costs, so that [the spell] will eventually resolve and have its effect. Casting a spell includes proposal of the spell (rules 601.2a-d) and determination and payment of costs (rules 601.2f-h). To cast a spell, a player follows the steps listed below, in order. A player must be legally allowed to cast the spell to begin this process (see rule 601.3). If a player is unable to comply with the requirements of a step listed below while performing that step, the casting of the spell is illegal; the game returns to the moment before the casting of that spell was proposed (see rule 723, "Handling Illegal Actions").
		// Rule 601.2a: To propose the casting of a spell, a player first [uses a card to create a spell and puts the spell on] the stack. [The spell] becomes the topmost object on the stack. [The spell] has all the characteristics of the card... associated with it, and [the casting] player becomes its controller. The spell remains on the stack until it's countered, it resolves, or an effect moves it elsewhere.
		// Rule 601.2e: The game checks to see if the proposed spell can legally be cast. If the proposed spell is illegal, the game returns to the moment before the casting of that spell was proposed (see rule 723, "Handling Illegal Actions").
		this.determines_playability_and_a_list_of_sufficient_combinations_of_configurations_of_a_permanent_to_contribute_mana_for_her_hand_cards();
		
		ArrayList<a_nonland_card> The_List_Of_Nonland_Cards_That_May_Be_Used_To_Cast_A_Spell = new ArrayList<a_nonland_card>();
		for (a_nonland_card The_Card : this.Hand.provides_its_list_of_nonland_cards()) {
			if (The_Card.is_playable()) {
				The_List_Of_Nonland_Cards_That_May_Be_Used_To_Cast_A_Spell.add(The_Card);
			}
		}
		System.out.println(this.Name + " may play cast a spell using a card in the following list. " + The_List_Of_Nonland_Cards_That_May_Be_Used_To_Cast_A_Spell);
		
		if (The_List_Of_Nonland_Cards_That_May_Be_Used_To_Cast_A_Spell.size() > 0) {
			a_nonland_card The_Nonland_Card_To_Use_To_Cast_A_Spell = this.chooses_a_card_to_use_to_cast_a_spell_from(The_List_Of_Nonland_Cards_That_May_Be_Used_To_Cast_A_Spell);
			//a_mana_pool The_Mana_Pool_To_Use_To_Cast_A_Spell =
			this.acquires_mana_for(The_Nonland_Card_To_Use_To_Cast_A_Spell);
			//this.Mana_Pool.increases_by(The_Mana_Pool_To_Use_To_Cast_A_Spell);
			//this.Mana_Pool.decreases_by(The_Mana_Pool_To_Use_To_Cast_A_Spell);
			The_Nonland_Card_To_Use_To_Cast_A_Spell = this.Hand.plays(The_Nonland_Card_To_Use_To_Cast_A_Spell);
			System.out.println("After playing a nonland card, the hand of " + this.Name + " has " + this.Hand.provides_its_number_of_cards() + " cards and contains the following. " + this.Hand);
			a_spell The_Spell = new a_spell(The_Nonland_Card_To_Use_To_Cast_A_Spell.provides_its_name(), The_Nonland_Card_To_Use_To_Cast_A_Spell.provides_its_type());
			this.Stack.receives(The_Spell);
			System.out.println("The stack contains the following spells. " + this.Stack);
		}
		
		while (this.Stack.contains_spells()) {
			a_spell The_Spell = this.Stack.provides_its_top_spell();
			if (The_Spell.provides_its_type().equals("Creature")) {
				this.Part_Of_The_Battlefield.receives_creature(new a_creature(The_Spell.provides_its_name(), false));
			}
		}
		System.out.println(this.Part_Of_The_Battlefield);
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 505.2: The main phase has no steps, so a main phase ends when all players pass in succession while the stack is empty.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire... Effects that last "until end of combat" expire at the end of the combat phase.
		
		this.Has_Priority = false;
	}
	
	
	public void determines_playability_and_a_list_of_sufficient_combinations_of_configurations_of_a_permanent_to_contribute_mana_for_her_hand_cards() {
		for (a_nonland_card The_Nonland_Card : this.Hand.provides_its_list_of_nonland_cards()) {
			this.determines_playability_and_a_list_of_sufficient_combinations_of_configurations_of_a_permanent_to_contribute_mana_for(The_Nonland_Card);
		}
	}
	
	
	public void determines_playability_and_a_list_of_sufficient_combinations_of_configurations_of_a_permanent_to_contribute_mana_for(a_nonland_card The_Nonland_Card) {
		The_Nonland_Card.becomes_not_playable();
		ArrayList<a_permanent> The_List_Of_Permanents = this.Part_Of_The_Battlefield.provides_its_list_of_permanents();
		ArrayList<a_configuration_of_a_permanent_to_contribute_mana> The_List_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana = new ArrayList<>();
		for (a_permanent The_Permanent : The_List_Of_Permanents) {
			for (int i : The_Permanent.provides_an_array_of_indices_of_available_options_for_contributing_mana()) {
				a_configuration_of_a_permanent_to_contribute_mana The_Available_Configuration_Of_A_Permanent_To_Contribute_Mana = new a_configuration_of_a_permanent_to_contribute_mana(The_Permanent, i);
				The_List_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana.add(The_Available_Configuration_Of_A_Permanent_To_Contribute_Mana);
			}
		}
		ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>> The_List_Of_Combinations_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana = this.generates_a_list_of_combinations_of_elements_in(The_List_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana);
		ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>> The_List_Of_Sufficient_Combinations_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana = new ArrayList<>();
		for (int i = 0; i < The_List_Of_Combinations_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana.size(); i++) {
			ArrayList<a_configuration_of_a_permanent_to_contribute_mana> The_Combination_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana = The_List_Of_Combinations_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana.get(i);
			a_mana_pool The_Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
			for (a_configuration_of_a_permanent_to_contribute_mana The_Available_Configuration_Of_A_Permanent_To_Contribute_Mana : The_Combination_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana) {
				a_permanent The_Permanent = The_Available_Configuration_Of_A_Permanent_To_Contribute_Mana.provides_its_permanent();
				int the_available_option_for_contributing_mana = The_Available_Configuration_Of_A_Permanent_To_Contribute_Mana.provides_its_option_for_contributing_mana();
				The_Mana_Pool.increases_by(The_Permanent.indicates_mana_pool_it_would_contribute_for(the_available_option_for_contributing_mana));
			}
			if (The_Mana_Pool.is_sufficient_for(The_Nonland_Card.provides_its_mana_cost())) {
				The_List_Of_Sufficient_Combinations_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana.add(The_Combination_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana);
			}
		}
		if (The_List_Of_Sufficient_Combinations_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana.size() > 0) {
			The_Nonland_Card.becomes_playable();
			The_Nonland_Card.receives(The_List_Of_Sufficient_Combinations_Of_Available_Configurations_Of_A_Permanent_To_Contribute_Mana);
		}
	}	
	
	public void completes_her_postcombat_main_phase() {
	
		System.out.println(this.Name + " is completing their postcombat main phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire... Effects that last "until end of combat" expire at the end of the combat phase.
	}
	
	
	public void completes_her_untap_step() {
		
		System.out.println(this.Name + " is completing their untap step.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 502.1: First, all phased-in permanents with phasing that the active player controls phase out, and all phased-out permanents that the active player controlled when they phased out phase in. This all happens simultaneously. This turn-based action doesn't use the stack. See rule 702.25, "Phasing."

		// Rule 502.2: Second, the active player determines which permanents they control will untap. Then they untap them all simultaneously. This turn-based action doesn't use the stack. Normally, all of a player's permanents untap, but effects can keep one or more of a player's permanents from untapping.
		this.determines_her_permanents_to_untap();
		this.untaps_her_permanents();
		
		// Rule 502.3: No player receives priority during the untap step, so no spells can be cast or resolve and no abilities can be activated or resolve. Any ability that triggers during this step will be held until the next time a player would receive priority, which is usually during the upkeep step (See rule 503, "Upkeep Step.")
		// Rule 500.3: A step in which no players receive priority ends when all specified actions that take place during that step are completed.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
	}
	
	
	// TODO: Allow for adding abilities that triggered during the untap stap and abilities that triggered at the beginning of the upkeep step to the stack.
	// TODO: Allow for processing the stack.
	public void completes_her_upkeep_step() {
		
		System.out.println(this.Name + " is completing their upkeep step.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 503.1a: Any abilities that triggered during the untap step and any abilities that triggered at the beginning of the upkeep [step] are put onto the stack before the active player gets priority; the order in which they triggered doesn't matter. (See rule 603, "Handling Triggered Abilities.")
		
		// Rule 503.1: The upkeep step has no turn-based actions. Once it begins, the active player gets priority. (See rule 117, "Timing and Priority.")
		this.Has_Priority = true;
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
		
		this.Has_Priority = false;
	}
	
	// TODO: Use discretion in determining permanents to untap.
	public void determines_her_permanents_to_untap() {
		
		System.out.println(this.Name + " is determining their permanents to untap.");
		
		this.List_Of_Permanents_That_Should_Be_Untapped.clear();
		
		for (an_artifact The_Artifact : this.Part_Of_The_Battlefield.provides_its_list_of_artifacts()) {
			if (The_Artifact.indicates_whether_it_is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Artifact);
			}
		}
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.provides_its_list_of_creatures()) {
			if (The_Creature.indicates_whether_it_is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Creature);
			}
		}
		for (an_enchantment The_Enchantment : this.Part_Of_The_Battlefield.provides_its_list_of_enchantments()) {
			if (The_Enchantment.indicates_whether_it_is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Enchantment);
			}
		}
		for (a_land The_Land : this.Part_Of_The_Battlefield.provides_its_list_of_lands()) {
			if (The_Land.indicates_whether_it_is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Land);
			}
		}
		for (a_planeswalker The_Planeswalker : this.Part_Of_The_Battlefield.provides_its_list_of_planeswalkers()) {
			if (The_Planeswalker.indicates_whether_it_is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Planeswalker);
			}
		}
	}
	
	
	public void draws() {
		this.Hand.receives(this.Deck.provides_a_card());
		
		System.out.println(
			"After drawing, the deck of " + this.Name + " has " + this.Deck.provides_its_number_of_cards() + " cards and contains the following. " + this.Deck
		);
		System.out.println(
			"After drawing, the hand of " + this.Name + " has " + this.Hand.provides_its_number_of_cards() + " cards and contains the following. " + this.Hand
		);
	}
	
	
	/** Rule 103.4:
	 * Each player draws a number of cards equal to their starting hand size, which is normally seven.
	 */
	
	public void draws_a_hand() {
		for (int i = 0; i < STARTING_HAND_SIZE; i++) {
			this.draws();
		}
	}
	
    private <T> ArrayList<ArrayList<T>> generates_a_list_of_combinations_of_elements_in(ArrayList<T> list_of_elements) {
        ArrayList<ArrayList<T>> list_of_combinations = new ArrayList<>();
        this.helps_generate_a_list_of_combinations(list_of_elements, 0, new ArrayList<T>(), list_of_combinations);
        return list_of_combinations;
    }

    private <T> void helps_generate_a_list_of_combinations(ArrayList<T> list_of_elements, int starting_index_in_list_of_elements, ArrayList<T> combination, ArrayList<ArrayList<T>> list_of_combinations) {
        list_of_combinations.add(new ArrayList<>(combination));
        for (int i = starting_index_in_list_of_elements; i < list_of_elements.size(); i++) {
            combination.add(list_of_elements.get(i));
            helps_generate_a_list_of_combinations(list_of_elements, i + 1, combination, list_of_combinations);
            combination.remove(combination.size() - 1);
        }
    }
	
	public void plays_a_land() {
		
		int The_Number_Of_Land_Cards = this.Hand.provides_its_number_of_land_cards();
		if (The_Number_Of_Land_Cards > 0) {
			System.out.println(this.Name + " is playing a land.");
			int The_Index_Of_The_Land_Card_To_Play = this.Random_Data_Generator.nextInt(0, The_Number_Of_Land_Cards - 1);
			a_land_card The_Land_Card_To_Play = this.Hand.provides_the_land_card_in_the_list_of_land_cards_at_index(The_Index_Of_The_Land_Card_To_Play);
			if (The_Land_Card_To_Play.provides_its_name().equals("Plains")) {
				this.Part_Of_The_Battlefield.receives_land(new a_Plains(false));
			} else if (The_Land_Card_To_Play.provides_its_name().equals("Forest")) {
				this.Part_Of_The_Battlefield.receives_land(new a_Forest(false));
			}
			System.out.println("After playing a land card, the hand of " + this.Name + " has " + this.Hand.provides_its_number_of_cards() + " cards and contains the following. " + this.Hand);
			System.out.println("After playing a land card, the part of the battlefield of " + this.Name + " has " + this.Part_Of_The_Battlefield.provides_its_number_of_permanents() + " cards and contains the following. " + this.Part_Of_The_Battlefield);
		}
	}
	
	public ArrayList<a_card> provides_a_list_of_cards_that_are_playable_given(ArrayList<a_mana_pool> The_List_Of_Possible_Mana_Pools) {
		
		ArrayList<a_card> The_List_Of_Cards_That_May_Be_Used_To_Cast_Spells = new ArrayList<a_card>();
		for (a_nonland_card The_Nonland_Card : this.Hand.provides_its_list_of_nonland_cards()) {
			for (a_mana_pool The_Possible_Mana_Pool : The_List_Of_Possible_Mana_Pools) {
				if (The_Possible_Mana_Pool.is_sufficient_for(The_Nonland_Card.provides_its_mana_cost())) {
					The_List_Of_Cards_That_May_Be_Used_To_Cast_Spells.add(The_Nonland_Card);
					//System.out.println("The mana pool of\n" + The_Possible_Mana_Pool + "\nis sufficient for the mana cost of " + The_Card.provides_its_name() + ",\n" + The_Card.provides_its_mana_cost());
					break;
				}
			}
		}
		
		return The_List_Of_Cards_That_May_Be_Used_To_Cast_Spells;
		
	}
	
	
	public String provides_her_name() {
		return this.Name;
	}
	
	public void receives(a_player The_Other_Player_To_Use) {
		this.Other_Player = The_Other_Player_To_Use;
	}
	
	public void shuffles_her_deck() {
		this.Deck.shuffles();
		System.out.println(
			"The deck of " + this.Name + " after shuffling has " + this.Deck.provides_its_number_of_cards() + " cards and is the following. " + this.Deck
		);
	}
	
	
	public void takes_her_turn() throws Exception {

		System.out.println(this.Name + " is taking their turn.");
		
		this.completes_her_beginning_phase();
		this.completes_her_precombat_main_phase();
		this.completes_her_combat_phase();
		this.completes_her_postcombat_main_phase();
		this.completes_her_end_phase();
		
		// Rule 500.5: Effects that last "until end of turn" are subject to special rules; see rule 514.2.	
	}
	
	
	public void untaps_her_permanents() {
		
		System.out.println(this.Name + " is untapping their permanents.");
		
		for (a_permanent The_Permanent : this.List_Of_Permanents_That_Should_Be_Untapped) {
			The_Permanent.untaps();
		}
	}

}
