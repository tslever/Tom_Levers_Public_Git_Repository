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
	
	
	public a_card chooses_a_card_to_use_to_cast_a_spell_from(ArrayList<a_card> The_List_Of_Cards_That_May_Be_Used_To_Cast_Spells) {
		
		int The_Index_Of_The_Card_To_Use = this.Random_Data_Generator.nextInt(0, The_List_Of_Cards_That_May_Be_Used_To_Cast_Spells.size());
		if (The_Index_Of_The_Card_To_Use != The_List_Of_Cards_That_May_Be_Used_To_Cast_Spells.size()) {
			a_card The_Card_To_Use = The_List_Of_Cards_That_May_Be_Used_To_Cast_Spells.remove(The_Index_Of_The_Card_To_Use);
			return The_Card_To_Use;
		}
		else {
			return null;
		}
		
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
			System.out.println("    Because " + this.Name + " is the starting player and this is the first turn, the draw step is skipped.");
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
	
	
	public void completes_her_precombat_main_phase() {
		
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
		this.determines_playability_mana_subpools_and_lists_of_mana_contributing_permanents_for_her_hand_cards();
		
		ArrayList<a_card> The_List_Of_Cards_That_May_Be_Used_To_Cast_A_Spell = new ArrayList<a_card>();
		for (a_card The_Card : this.Hand.provides_its_list_of_cards()) {
			if (The_Card.is_playable()) {
				The_List_Of_Cards_That_May_Be_Used_To_Cast_A_Spell.add(The_Card);
			}
		}
		System.out.println("List of cards that may be used to cast a spell: " + The_List_Of_Cards_That_May_Be_Used_To_Cast_A_Spell);
		
		a_card The_Card_To_Use_To_Cast_A_Spell = this.chooses_a_card_to_use_to_cast_a_spell_from(The_List_Of_Cards_That_May_Be_Used_To_Cast_A_Spell);
		if (The_Card_To_Use_To_Cast_A_Spell != null) {
			// Provide mana equal to the mana cost of the card.
			// Cast the card.
		}
		
		System.out.println("The stack contains the following spells.\n" + this.Stack);
		
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
	
	public void determines_playability_mana_subpools_and_lists_of_mana_contributing_permanents_for_her_hand_cards() {
		for (a_card The_Card : this.Hand.provides_its_list_of_cards()) {
			this.determines_playability_a_mana_subpool_to_be_used_and_a_list_of_mana_contributing_permanents_for(The_Card);
		}
	}
		
	public void determines_playability_a_mana_subpool_to_be_used_and_a_list_of_mana_contributing_permanents_for(a_card The_Card) {
		The_Card.becomes_not_playable();
		The_Card.nullifies_its_mana_subpool();
		The_Card.nullifies_its_list_of_mana_contributing_permanents();
		if (The_Card.provides_its_type().contains("Land")) {
			return;
		}
		ArrayList<a_mana_pool_and_a_list_of_permanents> The_List_Of_Objects_Of_Type_A_Mana_Pool_And_A_List_Of_Permanents = this.provides_a_list_of_objects_of_type_a_mana_pool_and_a_list_of_permanents();
		for (a_mana_pool_and_a_list_of_permanents The_Mana_Pool_And_The_List_Of_Permanents : The_List_Of_Objects_Of_Type_A_Mana_Pool_And_A_List_Of_Permanents) {
			a_mana_pool The_Mana_Subpool = The_Mana_Pool_And_The_List_Of_Permanents.provides_its_mana_pool();
			if (The_Mana_Subpool.is_sufficient_for(The_Card.provides_its_mana_cost())) {
				The_Card.becomes_playable();
				The_Card.receives(The_Mana_Subpool);
				The_Card.receives(The_Mana_Pool_And_The_List_Of_Permanents.provides_its_list_of_permanents());
				break;
			}
		}
	}
	
	public ArrayList<a_mana_pool_and_a_list_of_permanents> provides_a_list_of_objects_of_type_a_mana_pool_and_a_list_of_permanents() {
		ArrayList<a_mana_pool_and_a_list_of_permanents> The_List_Of_Objects_Of_Type_A_Mana_Pool_And_A_List_of_Permanents = new ArrayList<a_mana_pool_and_a_list_of_permanents>();
		ArrayList<a_permanent> The_List_Of_Permanents = this.Part_Of_The_Battlefield.provides_its_list_of_permanents();
		int The_Number_Of_Permanents = The_List_Of_Permanents.size();
		
		// Initialize an array of present indices, where each index represents a position in a permanent's array of mana contributions.
		int[] The_Array_Of_Present_Indices = new int[The_Number_Of_Permanents];
		for (int i = 0; i < The_Number_Of_Permanents; i++) {
			The_Array_Of_Present_Indices[i] = 0;
		}
		
		a_provider_of_an_array_of_possible_mana_contributions The_Provider_Of_An_Array_Of_Possible_Mana_Contributions = new a_provider_of_an_array_of_possible_mana_contributions();
		
		boolean A_Permanent_Has_More_Possible_Contributions = true;
		while (A_Permanent_Has_More_Possible_Contributions) {
			
			// Add to the list of objects of type a possible mana pool and a list of permanents a mana pool corresponding to the present mana contribution for each permanent and a list of these permanents.
			a_mana_pool The_Possible_Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
			ArrayList<a_permanent> The_List_Of_Contributing_Permanents = new ArrayList<a_permanent>();
			for (int i = 0; i < The_Number_Of_Permanents; i++) {
				a_permanent The_Permanent = The_List_Of_Permanents.get(i);
				a_mana_contribution[] The_Array_Of_Possible_Mana_Contributions = The_Provider_Of_An_Array_Of_Possible_Mana_Contributions.provides_the_array_of_possible_mana_contributions_corresponding_to(The_Permanent.provides_its_name());
				The_Possible_Mana_Pool.increases_by(The_Array_Of_Possible_Mana_Contributions[The_Array_Of_Present_Indices[i]]);
				The_List_Of_Contributing_Permanents.add(The_Permanent);
			}
			The_List_Of_Objects_Of_Type_A_Mana_Pool_And_A_List_of_Permanents.add(new a_mana_pool_and_a_list_of_permanents(The_Possible_Mana_Pool, The_List_Of_Contributing_Permanents));
						
			// Find the index of the right-most permanent with more possible mana contributions after the present mana contribution.
			int The_Index_Of_The_Right_Most_Permanent_With_More_Possible_Mana_Contributions = The_Number_Of_Permanents - 1;
			while ((The_Index_Of_The_Right_Most_Permanent_With_More_Possible_Mana_Contributions >= 0) && ((The_Array_Of_Present_Indices[The_Index_Of_The_Right_Most_Permanent_With_More_Possible_Mana_Contributions] + 1) >= The_Provider_Of_An_Array_Of_Possible_Mana_Contributions.provides_the_array_of_possible_mana_contributions_corresponding_to(The_List_Of_Permanents.get(The_Index_Of_The_Right_Most_Permanent_With_More_Possible_Mana_Contributions).provides_its_name()).length)) {
				The_Index_Of_The_Right_Most_Permanent_With_More_Possible_Mana_Contributions--;
			}
			
			// If no permanent has more possible contributions after the present mana contribution, stop looking for combinations of mana contributions.
			if (The_Index_Of_The_Right_Most_Permanent_With_More_Possible_Mana_Contributions < 0) {
				A_Permanent_Has_More_Possible_Contributions = false;
			}
			
			// Else; i.e., if there is a right-most permanent P with another possible mana contribution C, consider combinations of mana contributions with mana contribution C and all mana contributions of all permanents to the right of P.
			else {			
				The_Array_Of_Present_Indices[The_Index_Of_The_Right_Most_Permanent_With_More_Possible_Mana_Contributions] = The_Array_Of_Present_Indices[The_Index_Of_The_Right_Most_Permanent_With_More_Possible_Mana_Contributions] + 1;
				for (int i = The_Index_Of_The_Right_Most_Permanent_With_More_Possible_Mana_Contributions + 1; i < The_Number_Of_Permanents; i++) {
					The_Array_Of_Present_Indices[i] = 0;
				}
			}
			
		}
		
		return The_List_Of_Objects_Of_Type_A_Mana_Pool_And_A_List_of_Permanents;
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
		
		System.out.println("    " + this.Name + " is completing their untap step.");
		
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
		
		System.out.println("    " + this.Name + " is completing their upkeep step.");
		
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
		
		System.out.println("        " + this.Name + " is determining their permanents to untap.");
		
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
			"After drawing, the deck of " + this.Name + " has " + this.Deck.provides_its_number_of_cards() + " cards and contains the following.\n" +
			this.Deck + "\n"
		);
		System.out.println(
			"After drawing, the hand of " + this.Name + " has " + this.Hand.provides_its_number_of_cards() + " cards and contains the following.\n" +
			this.Hand + "\n"
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
	
	
	public void plays_a_land() {
		
		if (this.Hand.provides_its_number_of_land_cards() > 0) {
			System.out.println("    " + this.Name + " is playing a land.");
			int The_Index_Of_The_Land_Card_To_Play = this.Random_Data_Generator.nextInt(0, this.Hand.provides_its_number_of_land_cards() - 1);
			a_land_card The_Land_Card_To_Play = this.Hand.provides_the_land_card_at_index(The_Index_Of_The_Land_Card_To_Play);
			this.Part_Of_The_Battlefield.receives_land(new a_land(The_Land_Card_To_Play.provides_its_name(), false));
			System.out.println("After playing a land card, the hand of " + this.Name + " has " + this.Hand.provides_its_number_of_cards() + " cards and contains the following.\n" + this.Hand);
			System.out.println("After playing a land card, the part of the battlefield of " + this.Name + " has " + this.Part_Of_The_Battlefield.provides_its_number_of_permanents() + " cards and contains the following.\n" + this.Part_Of_The_Battlefield);
		}
	}
	
	public ArrayList<a_card> provides_a_list_of_cards_that_are_playable_given(ArrayList<a_mana_pool> The_List_Of_Possible_Mana_Pools) {
		
		ArrayList<a_card> The_List_Of_Cards_That_May_Be_Used_To_Cast_Spells = new ArrayList<a_card>();
		for (a_card The_Card : this.Hand.provides_its_list_of_nonland_cards()) {
			for (a_mana_pool The_Possible_Mana_Pool : The_List_Of_Possible_Mana_Pools) {
				if (The_Possible_Mana_Pool.is_sufficient_for(The_Card.provides_its_mana_cost())) {
					The_List_Of_Cards_That_May_Be_Used_To_Cast_Spells.add(The_Card);
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
	
	
	public void shuffles_her_deck() {
		this.Deck.shuffles();
		System.out.println(
			"The deck of " + this.Name + " after shuffling has " + this.Deck.provides_its_number_of_cards() + " cards and is the following.\n" +
			this.Deck + "\n"
		);
	}
	
	
	public void takes_her_turn() {

		System.out.println(this.Name + " is taking their turn.");
		
		this.completes_her_beginning_phase();
		this.completes_her_precombat_main_phase();
		this.completes_her_combat_phase();
		this.completes_her_postcombat_main_phase();
		this.completes_her_end_phase();
		
		// Rule 500.5: Effects that last "until end of turn" are subject to special rules; see rule 514.2.	
	}
	
	
	public void untaps_her_permanents() {
		
		System.out.println("        " + this.Name + " is untapping their permanents.");
		
		for (a_permanent The_Permanent : this.List_Of_Permanents_That_Should_Be_Untapped) {
			The_Permanent.untaps();
		}
	}

}
