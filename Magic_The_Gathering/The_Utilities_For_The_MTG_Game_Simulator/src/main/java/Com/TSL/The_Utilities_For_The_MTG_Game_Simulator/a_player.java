package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;
import java.util.Collections;

import org.apache.commons.math3.random.RandomDataGenerator;

public class a_player {
	
	/** Rule 103.4: Each player draws a number of cards equal to their starting hand size, which is normally seven. */
	private static int STARTING_HAND_SIZE = 7;
	
	private a_deck Deck;
	private an_exile Exile;
	private a_graveyard Graveyard;
	private a_hand Hand;
	private boolean Has_Cast_A_Spell_Or_Activated_A_Nonmana_Activated_Ability;
	private boolean Has_Passed;
	private boolean Has_Played_A_Land;
	private boolean Has_Played_A_Land_This_Turn;
	private boolean Has_Performed_A_State_Based_Action;
	private boolean Has_Priority;
	private boolean Has_Taken_An_Action;
	private int Index_Of_The_Present_Turn;
	private int Life;
	private ArrayList<a_creature> List_Of_Attackers;
	private ArrayList<a_battle> List_Of_Battles;
	private ArrayList<a_creature> List_Of_Blockers;
	private a_mana_pool Mana_Pool;
	private String Name;
	private a_part_of_the_battlefield Part_Of_The_Battlefield;
	private a_player Other_Player;
	private RandomDataGenerator Random_Data_Generator;
	private a_stack Stack;
	private boolean Was_Starting_Player;
	
	/**
	 * a_player
	 * 
	 * Rule 103.3: Each player begins the game with a starting life total of 20.
	 */
	public a_player(a_deck The_Deck_To_Use, String The_Name_To_Use, a_stack The_Stack_To_Use)
	{
		this.Deck = The_Deck_To_Use;
		this.Exile = new an_exile();
		this.Graveyard = new a_graveyard();
		this.Hand = new a_hand();
		this.Life = 20;
		this.List_Of_Attackers = new ArrayList<a_creature>();
		this.List_Of_Battles = new ArrayList<a_battle>();
		this.List_Of_Blockers = new ArrayList<a_creature>();
		this.Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
		this.Name = The_Name_To_Use;
		this.Part_Of_The_Battlefield = new a_part_of_the_battlefield();
		this.Random_Data_Generator = new RandomDataGenerator();
		this.Stack = The_Stack_To_Use;
	}

	public a_mana_pool acquires_mana_for_nonland_card_or_nonmana_activated_ability(Object The_Object) throws Exception {
		a_nonland_card The_Nonland_Card = null;
		a_nonmana_activated_ability The_Nonmana_Activated_Ability = null;
		if (The_Object instanceof a_nonland_card) {
			The_Nonland_Card = (a_nonland_card) The_Object;
		} else if (The_Object instanceof a_nonmana_activated_ability) {
			The_Nonmana_Activated_Ability = (a_nonmana_activated_ability) The_Object;
		} else {
			throw new Exception("Mana is acquired only for a nonland card or a nonmana activated ability.");
		}
		a_mana_pool The_Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
		ArrayList<ArrayList<a_mana_ability>> The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object;
		if (The_Nonland_Card != null) {
			The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object = The_Nonland_Card.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this();
		} else if (The_Nonmana_Activated_Ability != null) {
			The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object = The_Nonmana_Activated_Ability.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this();			
		} else {
			throw new Exception("A nonland card and a nonmana activated ability were null.");
		}
		ArrayList<a_mana_ability> The_Combination_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object = this.chooses_a_combination_of_available_mana_abilities_sufficient_to_play_an_object_from(The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object);
		for (a_mana_ability The_Available_Mana_Ability : The_Combination_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object) {
			The_Mana_Pool.increases_by(The_Available_Mana_Ability.activates_and_contributes_a_mana_pool());
		}
		return The_Mana_Pool;
	}
	
	public void assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_card_or_nonmana_activated_ability(Object The_Object) throws Exception {
		a_nonland_card The_Nonland_Card = null;
		a_nonmana_activated_ability The_Nonmana_Activated_Ability = null;
		if (The_Object instanceof a_nonland_card) {
			The_Nonland_Card = (a_nonland_card) The_Object;
		} else if (The_Object instanceof a_nonmana_activated_ability) {
			The_Nonmana_Activated_Ability = (a_nonmana_activated_ability) The_Object;
		} else {
			throw new Exception("A list of sufficient combinations of available mana abilities is assigned only to a nonland card or nonmana activated ability.");
		}
		ArrayList<a_permanent> The_List_Of_Permanents = this.Part_Of_The_Battlefield.list_of_permanents();
		ArrayList<a_mana_ability> The_List_Of_Available_Mana_Abilities_For_The_Player = new ArrayList<>();
		for (a_permanent The_Permanent : The_List_Of_Permanents) {
			ArrayList<a_mana_ability> The_List_Of_Available_Mana_Abilities_For_The_Permanent = The_Permanent.list_of_available_mana_abilities();
			The_List_Of_Available_Mana_Abilities_For_The_Player.addAll(The_List_Of_Available_Mana_Abilities_For_The_Permanent);
		}
		ArrayList<ArrayList<a_mana_ability>> The_List_Of_Combinations_Of_Available_Mana_Abilities = this.generates_a_list_of_combinations_of_elements_in(The_List_Of_Available_Mana_Abilities_For_The_Player);
		ArrayList<ArrayList<a_mana_ability>> The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities = new ArrayList<>();
		for (ArrayList<a_mana_ability> The_Combination_Of_Available_Mana_Abilities : The_List_Of_Combinations_Of_Available_Mana_Abilities) {
			a_mana_pool The_Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
			for (a_mana_ability The_Mana_Ability : The_Combination_Of_Available_Mana_Abilities) {
				The_Mana_Pool.increases_by(The_Mana_Ability.indicates_the_mana_pool_it_would_contribute());
			}
			a_mana_cost The_Mana_Cost = null;
			if (The_Nonland_Card != null) {
				The_Mana_Cost = The_Nonland_Card.mana_cost();
			} else if (The_Nonmana_Activated_Ability != null) {
				The_Mana_Cost = The_Nonmana_Activated_Ability.mana_cost();
			} else {
				throw new Exception("A nonland card and a nonmana activated ability were null.");
			}
			if (The_Mana_Cost != null) {
				if (The_Mana_Pool.is_sufficient_for(The_Mana_Cost)) {
					The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities.add(The_Combination_Of_Available_Mana_Abilities);
				}
			} else {
				throw new Exception("A mana cost was null.");
			}
		}
		if (The_Nonland_Card != null) {
			The_Nonland_Card.receives(The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities);
		} else if (The_Nonmana_Activated_Ability != null) {
			The_Nonmana_Activated_Ability.receives(The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities);
		} else {
			throw new Exception("A nonland card and a nonmana activated ability were null.");
		}
	}
	
	public void assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_hand_cards_in(ArrayList<a_nonland_card> The_List_Of_Nonland_Hand_Cards) throws Exception {
		for (a_nonland_card The_Nonland_Hand_Card : The_List_Of_Nonland_Hand_Cards) {
			this.assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_card_or_nonmana_activated_ability(The_Nonland_Hand_Card);
		}
	}
	
	public void assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_her_permanents_nonmana_activated_abilities() throws Exception {
		for (a_permanent The_Permanent : this.Part_Of_The_Battlefield.list_of_permanents()) {
			for (a_nonmana_activated_ability The_Nonmana_Activated_Ability : The_Permanent.list_of_nonmana_activated_abilities()) {
				this.assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_card_or_nonmana_activated_ability(The_Nonmana_Activated_Ability);
			}
		}
	}
	
	public void becomes_the_starting_player() {
		this.Was_Starting_Player = true;
	}
	
	
	public Object chooses_a_playable_nonland_card_or_an_activatable_nonmana_activated_ability_from(ArrayList<Object> The_List_Of_Playable_Nonland_Cards_And_Activatable_Nonmana_Activated_Abilities) {
		int The_Random_Index_Of_A_Playable_Nonland_Card_Or_An_Activatable_Nonmana_Activated_Ability_In_The_List = this.Random_Data_Generator.nextInt(0, The_List_Of_Playable_Nonland_Cards_And_Activatable_Nonmana_Activated_Abilities.size() - 1);
		Object The_Random_Playable_Nonland_Card_Or_Activatable_Nonmana_Activated_Ability = The_List_Of_Playable_Nonland_Cards_And_Activatable_Nonmana_Activated_Abilities.remove(The_Random_Index_Of_A_Playable_Nonland_Card_Or_An_Activatable_Nonmana_Activated_Ability_In_The_List);
		return The_Random_Playable_Nonland_Card_Or_Activatable_Nonmana_Activated_Ability;
	}
	
	
	public ArrayList<a_mana_ability> chooses_a_combination_of_available_mana_abilities_sufficient_to_play_an_object_from(ArrayList<ArrayList<a_mana_ability>> The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_An_Object) {
		int The_Index_Of_The_Sufficient_Combination = this.Random_Data_Generator.nextInt(0, The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_An_Object.size() - 1);
		ArrayList<a_mana_ability> The_Sufficient_Combination = The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_An_Object.get(The_Index_Of_The_Sufficient_Combination);
		return The_Sufficient_Combination;
	}
	
	 /* Rule 501.1: The beginning phase consists of three steps, in this order: untap, upkeep, and draw. */
	public void completes_her_beginning_phase() throws Exception {
		System.out.println(this + " is completing " + this + "'s Beginning Phase.");
		this.Has_Played_A_Land_This_Turn = false;
		 /* No player receives priority during the untap step.
		 * Rule 503.1a: Any abilities that triggered during the untap step and any abilities that triggered at the beginning of the upkeep [step] are put onto the stack before the active player gets priority; the order in which they triggered doesn't matter. (See rule 603, "Handling Triggered Abilities.") */
		this.completes_her_untap_step();
		this.completes_her_upkeep_step();
		/* Rule 103.7a: In a two-player game, the player who plays first skips the draw step (see rule 504, "Draw Step") of their first turn. */
		if (this.Was_Starting_Player && this.Index_Of_The_Present_Turn == 0) {
			System.out.println("Because " + this + " is the starting player and " + this + " is taking " + this + "'s first turn, " + this + " skips " + this + "'s draw step.");
		} else {
			this.completes_her_draw_step();
		}
		/* Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		 * Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		 * Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire. */
	}
	
	/**
	 * completes_her_beginning_of_combat_step
	 * 
	 * Rule 507.2: Second, the active player gets priority. (See [R]ule 117, "Timing and Priority.")
	 */
	public void completes_her_beginning_of_combat_step() throws Exception {
		this.performs_state_based_actions_adds_trigged_abilities_to_stack_and_has_this_and_other_player_cast_spells_activate_abiltiies_and_take_special_actions("Beginning Of Combat Step");
	}
	
	/**
	 * completes_her_declare_attackers_step
	 * 
	 * Rule 508.1: First, the active player declares attackers. This turn-based action doesn't use the stack.
	 * To declare attackers, the active player follows the steps below, in order.
	 * If at any point during the declaration of attackers, the active player is unable to comply with any of steps listed below, the declaration is illegal; the game returns to the moment before the declaration (see [R]ule 728, "Handling Illegal Actions").
	 * Rule 508.1a: The active player chooses which creatures that they control, if any, will attack.
	 * The creatures must be untapped, they can't also be battles, and each one must either have haste or have been controlled by the active player continuously since the turn began.
	 * Rule 508.1b: If the defending player controls any planeswalkers, is the protector of any battles, or the game allows the active player to attack multiple other players, the active player announces which player, planewalker, or battle each of the chosen creatures is attacking.
	 * Rule 508.1c: The active player checks each creature they control to see whether [the creature is] affected by any restrictions (effects that say a creature can't attack, or that [the creature] can't attack unless some condition is met).
	 * If any restrictions are being disobeyed, the declaration of attackers is illegal.
	 * Rule 508.1c: The active player checks each creature they control to see whether [the creature is] affected by any requirements (effects that say attacks if able, or that it attacks if some condition is met).
	 * If the number of requirements that are being obeyed is fewer than the maximum possible number of requirements that could be obeyed without disobeying any restrictions, the declaration of attackers is illegal.
	 * If a creature can't attack unless a player pays a cost, that player is not required to pay that cost, even if attacking with that creature would increase the number of requirements being obeyed.
	 * If a requirement that says a creature attacks if able during a certain turn refers to a turn with multiple combat phases, the creature attacks if able during each declare attackers step in that turn.
	 * Rule 508.1e: If any of the chosen creatures have banding or a "bands with other" ability, the active player announces which creatures, if any, are banded with which. (See [R]ule 702.22, "Banding.")
	 * Rule 508.1f: The active player taps the chosen creatures. Tapping a creature when [the creature is] declared as an attacker isn't a cost; attacking simply causes creatures to become tapped.
	 * @throws Exception 
	 */
	public void completes_her_declare_attackers_step() throws Exception {
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			if (!The_Creature.is_tapped() && !The_Creature.is_battle() && (The_Creature.has_haste() || The_Creature.has_been_controlled_by_the_active_player_continuously_since_the_turn_began()) && The_Creature.can_attack()) {
				if (The_Creature.must_attack() || (an_enumeration_of_states_of_a_coin.provides_a_state() == an_enumeration_of_states_of_a_coin.HEADS)) {
					The_Creature.sets_its_indicator_of_whether_it_is_attacking_to(true);
					Object The_Attackee = null;
					if (!this.Part_Of_The_Battlefield.list_of_planeswalkers().isEmpty() || !this.List_Of_Battles.isEmpty()) {
						ArrayList<Object> The_List_Of_Possible_Attackees = new ArrayList<>();
						The_List_Of_Possible_Attackees.add(this);
						for (a_planeswalker The_Planeswalker : this.Other_Player.Part_Of_The_Battlefield.list_of_planeswalkers()) {
							The_List_Of_Possible_Attackees.add(The_Planeswalker);
						}
						for (a_battle The_Battle : this.List_Of_Battles) {
							The_List_Of_Possible_Attackees.add(The_Battle);
						}
						int The_Index_Of_The_Attackee = this.Random_Data_Generator.nextInt(0, The_List_Of_Possible_Attackees.size() - 1);
						The_Attackee = The_List_Of_Possible_Attackees.get(The_Index_Of_The_Attackee);
					} else {
						The_Attackee = this.Other_Player;
					}
					if (The_Attackee != null) {
						The_Creature.attacks(The_Attackee);
						System.out.println("The creature " + The_Creature + " attacks " + The_Attackee + ".");
					} else {
						throw new Exception("The target is null");
					}
					The_Creature.taps();
					this.List_Of_Attackers.add(The_Creature);
				}
			}
		}
		this.performs_state_based_actions_adds_trigged_abilities_to_stack_and_has_this_and_other_player_cast_spells_activate_abiltiies_and_take_special_actions("Declare Attackers Step");
	}
	
	public void declares_blockers() throws Exception {
		this.Has_Priority = true;
		ArrayList<a_creature> The_List_Of_Blockers = new ArrayList<>();
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			if (!The_Creature.is_tapped() && !The_Creature.is_battle()) {
				if (The_Creature.must_block() || (an_enumeration_of_states_of_a_coin.provides_a_state() == an_enumeration_of_states_of_a_coin.HEADS)) {
					The_List_Of_Blockers.add(The_Creature);
				}
			}
		}
		for (a_creature The_Blocker : The_List_Of_Blockers) {
			ArrayList<a_creature> The_List_Of_Attackers_That_The_Blocker_Can_Block = new ArrayList<>();
			for (a_creature The_Attacker : this.List_Of_Attackers) {
				if (The_Blocker.can_block(The_Attacker)) {
					The_List_Of_Attackers_That_The_Blocker_Can_Block.add(The_Blocker);
				}
			}
			int The_Index_Of_Attacker = this.Random_Data_Generator.nextInt(0, The_List_Of_Attackers_That_The_Blocker_Can_Block.size() - 1);
			a_creature The_Attacker = The_List_Of_Attackers_That_The_Blocker_Can_Block.get(The_Index_Of_Attacker);
			The_Blocker.blocks(The_Attacker);
			The_Attacker.becomes_blocked_by(The_Blocker);
		}
		this.Has_Priority = false;
	}
	
	public void chooses_damage_assignment_order_for_her_attackers() {
		for (a_creature The_Attacker : this.List_Of_Attackers) {
			if (The_Attacker.is_blocked()) {
				Collections.shuffle(The_Attacker.list_of_blockers());
			}
		}
	}
	
	public void chooses_damage_assignment_order_for_her_blockers() {
		this.Has_Priority = true;
		for (a_creature The_Blocker : this.List_Of_Blockers) {
			Collections.shuffle(The_Blocker.list_of_blockees());
		}
		this.Has_Priority = false;
	}
	
	/**
	 * completes_her_declare_blockers_step
	 * 
	 * Rule 509.1: First, the defending player declares blockers. This turn-based action doesn't use the stack. To declare blockers, the defending player follows the steps below, in order.
	 * If at any point during the declaration of blockers, the defending player is unable to comply with any of the steps listed below, the declaration is illegal; the game returns to the moment before the declaration (see [R]ule 728, "Handling Illegal Actions").
	 * Rule 509.1a: The defending player chooses which creatures they control, if any, will block.
	 * The chosen creatures must be untapped and they can't also be battles.
	 * For each of the chosen creatures, the defending player chooses one creature for [the chosen creature] to block that is attacking [the defending] player, a planeswalker [the defending player] control[s], or a battle [the defending player] protect[s].
	 */
	public void completes_her_declare_blockers_step() throws Exception {
		this.Other_Player.declares_blockers();
		this.chooses_damage_assignment_order_for_her_attackers();
		this.Other_Player.chooses_damage_assignment_order_for_her_blockers();
		
	}
	
	public void has_her_attackers_assign_combat_damage() throws Exception {
		for (a_creature The_Attacker : this.List_Of_Attackers) {
			if (The_Attacker.is_blocked()) {
				ArrayList<a_creature> The_List_Of_Blockers = The_Attacker.list_of_blockers();
				int combat_damage_to_be_dealt = The_Attacker.effective_power();
				for (a_creature The_Blocker : The_List_Of_Blockers) {
					if (combat_damage_to_be_dealt <= The_Blocker.effective_toughness()) {
						The_Blocker.receives_combat_damage(combat_damage_to_be_dealt);
					} else {
						The_Blocker.receives_combat_damage(The_Blocker.effective_toughness());
						combat_damage_to_be_dealt -= The_Blocker.effective_toughness();
					}
				}
			} else {
				Object The_Attackee = The_Attacker.attackee();
				if (The_Attackee instanceof a_player) {
					a_player The_Player = (a_player) The_Attackee;
					The_Player.Life -= The_Attacker.effective_power();
				} else if (The_Attackee instanceof a_planeswalker) {
					a_planeswalker The_Planeswalker = (a_planeswalker) The_Attackee;
					The_Planeswalker.receives_combat_damage(The_Attacker.effective_power());
				} else if (The_Attackee instanceof a_battle) {
					throw new Exception("Not implemented");
				}
			}
		}
	}
	
	private void puts_cards_corresponding_to_creatures_dealt_lethal_damage_in_graveyard() {
		ArrayList<a_creature> The_List_Of_Creatures = new ArrayList<>();
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			if (The_Creature.was_dealt_lethal_damage()) {
				The_List_Of_Creatures.add(The_Creature);
			}
		}
		this.Part_Of_The_Battlefield.list_of_creatures().removeAll(The_List_Of_Creatures);
		for (a_creature The_Creature : The_List_Of_Creatures) {
			this.Graveyard.receives(The_Creature.creature_card());	
		}
	}
	
	public void has_her_blockers_assign_combat_damage() {
		this.Has_Priority = true;
		for (a_creature The_Blocker : this.List_Of_Blockers) {
			ArrayList<a_creature> The_List_Of_Blockees = The_Blocker.list_of_blockees();
			int combat_damage_to_be_dealt = The_Blocker.effective_power();
			for (a_creature The_Blockee : The_List_Of_Blockees) {
				if (combat_damage_to_be_dealt <= The_Blockee.effective_toughness()) {
					The_Blockee.receives_combat_damage(combat_damage_to_be_dealt);
				} else {
					The_Blockee.receives_combat_damage(The_Blockee.effective_toughness());
					combat_damage_to_be_dealt -= The_Blockee.effective_toughness();
				}
			}
		}
		this.Has_Priority = false;
	}
	
	/**
	 * completes_her_combat_damage_step
	 */
	public void completes_her_combat_damage_step() throws Exception {
		this.has_her_attackers_assign_combat_damage();
		this.Other_Player.has_her_blockers_assign_combat_damage();
		this.puts_cards_corresponding_to_creatures_dealt_lethal_damage_in_graveyard();
		this.Other_Player.puts_cards_corresponding_to_creatures_dealt_lethal_damage_in_graveyard();
	}
	
	/**
	 * completes_her_end_of_combat_step
	 */
	public void completes_her_end_of_combat_step() {
		this.Has_Priority = true;
	}
	
	/**
	 * completes_her_combat_phase
	 * 
	 * Rule 506.2: During the combat phase, the active player is the attacking player; creatures that player controls may attack.
	 * During the combat phase of a two-player game, the nonactive player is the defending player; that player, planeswalkers they control, and battles they protect may be attacked.
	 * Rule 506.3a: Only a creature can attack or block. Only a player, a planeswalker, or a battle can be attacked.
	 * Rule 506.5: A creature attacks alone if it's the only creature declared as an attacker during the declare attackers step.
	 * A creature blocks alone if it's the only creature declared as a blocker during the declare blockers step.
	 * A creature is blocking alone if it's blocking but no other creatures are.
	 * @throws Exception 
 	 */
	public void completes_her_combat_phase() throws Exception {
		
		System.out.println(this.Name + " is completing their combat phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 506.1: The combat phase has five steps, which proceed in order: beginning of combat, declare attackers, declare blockers, combat damage, and end of combat.
		// The declare blockers and combat damage steps are skipped if no creatures are declared as attackers or put onto the battlefield attacking (see rule 508.8).
		// There are two combat damage steps if any attacking or blocking creature has first strike (see 702.7) or double strike (see 702.4).
		this.completes_her_beginning_of_combat_step();
		this.completes_her_declare_attackers_step();
		if (!this.List_Of_Attackers.isEmpty()) {
			this.completes_her_declare_blockers_step();
			this.completes_her_combat_damage_step();
		}
		
		int number_of_combat_damage_steps = 1;
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			ArrayList<a_static_ability> The_List_Of_Static_Abilities = The_Creature.list_of_static_abilities();
			for (a_static_ability The_Static_Ability : The_List_Of_Static_Abilities) {
				if (The_Static_Ability.effect().equals("first strike") || The_Static_Ability.effect().equals("double strike")) {
					number_of_combat_damage_steps = 2;
				}
			}
		}
		for (int i = 0; i < number_of_combat_damage_steps; i++) {
			this.completes_her_combat_damage_step();
		}
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire... Effects that last "until end of combat" expire at the end of the combat phase.
	}
	
	
    public void completes_her_draw_step() throws Exception {
		System.out.println(this + " is completing " + this + "'s draw step.");
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		// Rule 504.1: First, the active player draws a card. This turn-based action doesn't use the stack.
		this.draws();
		
		// Rule 504.2: Second, the active player gets priority. (See rule 117, "Timing and Priority.")
		this.performs_state_based_actions_adds_trigged_abilities_to_stack_and_has_this_and_other_player_cast_spells_activate_abiltiies_and_take_special_actions("Draw Step");
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
	}
	
	
	public void completes_her_end_phase() {
		
		System.out.println(this.Name + " is completing their end phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire... Effects that last "until end of combat" expire at the end of the combat phase.
		
	}
	
	public void casts_a_spell_or_activates_a_nonmana_activated_ability(String The_Step_To_Use, boolean Indicator_Of_Whether_This_Player_May_Only_Play_Instants_Nonland_Hand_Cards_With_Flash_And_Nonmana_Activated_Abilities) throws Exception {
		// Rule 601.2: To cast a spell is to [use a card to create a spell], put [the spell] on the stack, and pay its mana costs, so that [the spell] will eventually resolve and have its effect. Casting a spell includes proposal of the spell (rules 601.2a-d) and determination and payment of costs (rules 601.2f-h). To cast a spell, a player follows the steps listed below, in order. A player must be legally allowed to cast the spell to begin this process (see rule 601.3). If a player is unable to comply with the requirements of a step listed below while performing that step, the casting of the spell is illegal; the game returns to the moment before the casting of that spell was proposed (see rule 723, "Handling Illegal Actions").
		// Rule 601.2e: The game checks to see if the proposed spell can legally be cast. If the proposed spell is illegal, the game returns to the moment before the casting of that spell was proposed (see rule 723, "Handling Illegal Actions").
		System.out.println(this + " is considering casting a spell or activating an ability.");
		ArrayList<a_nonland_card> The_List_Of_All_Nonland_Hand_Cards = this.Hand.list_of_nonland_cards();
		ArrayList<a_nonland_card> The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step;
		if (Indicator_Of_Whether_This_Player_May_Only_Play_Instants_Nonland_Hand_Cards_With_Flash_And_Nonmana_Activated_Abilities) {
			The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step = new ArrayList<>();
			for (a_nonland_card The_Nonland_Hand_Card : The_List_Of_All_Nonland_Hand_Cards) {
				if (The_Nonland_Hand_Card.type().equals("Instant") || The_Nonland_Hand_Card.text().contains("Flash")) {
					The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step.add(The_Nonland_Hand_Card);
				}
			}
		} else {			
			The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step = The_List_Of_All_Nonland_Hand_Cards;
		}
		this.assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_hand_cards_in(The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step);
		this.assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_her_permanents_nonmana_activated_abilities();
		this.determines_whether_are_playable_the_cards_in(The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step, The_Step_To_Use);
		this.determines_whether_her_permanents_nonmana_activated_abilities_are_activatable();
		ArrayList<a_nonland_card> The_List_Of_Playable_Nonland_Hand_Cards = this.generates_a_list_of_playable_nonland_hand_cards_in(The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step);
		System.out.println(this.Name + " may cast a spell using a card in the following list. " + The_List_Of_Playable_Nonland_Hand_Cards);
		ArrayList<a_nonmana_activated_ability> The_List_Of_Activatable_Nonmana_Activated_Abilities = this.generates_a_list_of_activatable_nonmana_activated_abilities();
		System.out.println(this.Name + " may activate an ability in the following list. " + The_List_Of_Activatable_Nonmana_Activated_Abilities);
		ArrayList<Object> The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities = new ArrayList<>();
		The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities.addAll(The_List_Of_Playable_Nonland_Hand_Cards);
		The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities.addAll(The_List_Of_Activatable_Nonmana_Activated_Abilities);
		
		// Rule 601.2a: To propose the casting of a spell, a player first [uses a card to create a spell and puts the spell on] the stack. [The spell] becomes the topmost object on the stack. [The spell] has all the characteristics of the card... associated with it, and [the casting] player becomes its controller. The spell remains on the stack until it's countered, it resolves, or an effect moves it elsewhere.
		if (!The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities.isEmpty()) {
			Object The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability = this.chooses_a_playable_nonland_card_or_an_activatable_nonmana_activated_ability_from(The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities);
			/* Rule 117.1d: A player may activate a mana ability whenever they have priority, whenever they are casting a spell or activating an ability that requires a mana payment, or whenever a rule or effect acts for a mana payment (even in the middle of casting or resolving a spell or activating or resolving an ability). */
			//a_mana_pool The_Mana_Pool_To_Use_To_Cast_A_Spell =
			this.acquires_mana_for_nonland_card_or_nonmana_activated_ability(The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability);
			//this.Mana_Pool.increases_by(The_Mana_Pool_To_Use_To_Cast_A_Spell);
			//this.Mana_Pool.decreases_by(The_Mana_Pool_To_Use_To_Cast_A_Spell);
			// Rule 117.3c: If a player has priority when they cast a spell, activate an ability, or take a special action, that player receives priority afterward.
			if (The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability instanceof a_nonland_card) {
				a_nonland_card The_Playable_Nonland_Hand_Card = (a_nonland_card) The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability;
				System.out.println(this + " plays the nonland hand card " + The_Playable_Nonland_Hand_Card + ".");
				this.Hand.removes(The_Playable_Nonland_Hand_Card);
				System.out.println("After playing the nonland hand card " + The_Playable_Nonland_Hand_Card + ", " + this + "'s hand has " + this.Hand.number_of_cards() + " cards and contains the following. " + this.Hand);
				String The_Type_Of_The_Playable_Nonland_Hand_Card = The_Playable_Nonland_Hand_Card.type();
				if (The_Type_Of_The_Playable_Nonland_Hand_Card.equals("Instant") || The_Type_Of_The_Playable_Nonland_Hand_Card.equals("Sorcery")) {
				    a_spell The_Spell = new a_spell(The_Playable_Nonland_Hand_Card, this);
				    this.Stack.receives(The_Spell);
				    System.out.println(this + " has cast instant or sorcery spell " + The_Spell + ".");
				} else {
					a_permanent_spell The_Permanent_Spell = new a_permanent_spell(The_Playable_Nonland_Hand_Card, this);
					this.Stack.receives(The_Permanent_Spell);
				    System.out.println(this + " has cast permanent spell " + The_Permanent_Spell + ".");
				}
				this.Has_Passed = false;
				this.Has_Taken_An_Action = true;
				this.Has_Cast_A_Spell_Or_Activated_A_Nonmana_Activated_Ability = true;
				System.out.println("The stack contains the following spells and abilities. " + this.Stack);
			} else if (The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability instanceof a_nonmana_activated_ability) {
				a_nonmana_activated_ability The_Nonmana_Activated_Ability = (a_nonmana_activated_ability) The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability;
				System.out.println(" activates the ability " + The_Nonmana_Activated_Ability);
				this.Stack.receives(The_Nonmana_Activated_Ability);
				System.out.println(this.Name + " has activated ability " + The_Nonmana_Activated_Ability + ".");
				this.Has_Passed = false;
				this.Has_Taken_An_Action = true;
				this.Has_Cast_A_Spell_Or_Activated_A_Nonmana_Activated_Ability = true;
				System.out.println("The stack contains the following spells and abilities. " + this.Stack);
			}
		} else {
			this.Has_Passed = true;
			this.Has_Taken_An_Action = false;
			this.Has_Cast_A_Spell_Or_Activated_A_Nonmana_Activated_Ability = false;
			System.out.println(this + " neither casts a spell nor activates an ability.");
		}
	}
	
	public void completes_a_main_phase() throws Exception {
		System.out.println(this.Name + " is starting a main phase.");
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			The_Creature.sets_its_indicator_of_whether_it_has_been_controlled_by_the_active_player_continuously_since_the_turn_began_to(true);
		}
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 505.4: Second, if the active player controls one or more Saga enchantments and it's the active player's precombat main phase, the active player puts a lore counter on each Saga they control. (See rule 714, "Saga Cards.") This turn-based action doesn't use the stack.

		// Rule 505.5: Third, the active player gets priority. (See rule 117, "Timing and Priority.")
		this.performs_state_based_actions_adds_trigged_abilities_to_stack_and_has_this_and_other_player_cast_spells_activate_abiltiies_and_take_special_actions("Main Phase");
	}
	
	
	public void completes_her_untap_step() {
		System.out.println(this + " is completing " + this + "'s Untap Step.");
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		// Rule 502.1: First, all phased-in permanents with phasing that the active player controls phase out, and all phased-out permanents that the active player controlled when they phased out phase in. This all happens simultaneously. This turn-based action doesn't use the stack. See rule 702.25, "Phasing."
		// TODO
		// Rule 502.2: Second, the active player determines which permanents they control will untap. Then they untap them all simultaneously. This turn-based action doesn't use the stack. Normally, all of a player's permanents untap, but effects can keep one or more of a player's permanents from untapping.
		// Rule 502.3: No player receives priority during the untap step, so no spells can be cast or resolve and no abilities can be activated or resolve. Any ability that triggers during this step will be held until the next time a player would receive priority, which is usually during the upkeep step (See rule 503, "Upkeep Step.")
		ArrayList<a_permanent> The_List_Of_Permanents_That_Should_Be_Untapped = this.determines_her_permanents_to_untap();
		if (The_List_Of_Permanents_That_Should_Be_Untapped.isEmpty()) {
			System.out.println("No permanents are untapped.");
		} else {
			this.untaps_her_permanents(The_List_Of_Permanents_That_Should_Be_Untapped);
			System.out.println("Permanents in the above list are untapped.");
		}
		// Rule 500.3: A step in which no players receive priority ends when all specified actions that take place during that step are completed.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
	}
	
	public void performs_all_applicable_state_based_actions_as_a_single_event() {
		// TODO
	}
	
	public void performs_state_based_actions_adds_trigged_abilities_to_stack_and_has_this_and_other_player_cast_spells_activate_abiltiies_and_take_special_actions(String The_Step_To_Use) throws Exception {
		/* Rule 117.3c: If a player has priority when they cast a spell, activate an ability, or take a special action, that player receives priority afterward.
		 * If a player has priority and chooses not to take any actions, that player passes.
		 * If any mana is in that player's mana pool, they announce what mana is there.
		 * Then the next player in turn order receives priority.
		 * Rule 117.4: If all players pass in succession (that is, if all players pass without taking any actions in between passing), the spell or ability on top of the stack resolves or, if the stack is empty, the phase or step ends. */
		do {
			do {
				/* Rule 117.5: Each time a player would get priority, the game first performs all applicable state-based actions as a single event (see rule 704, "State-Based Actions"), then repeats this process until no state-based actions are performed.
				 * Then triggered abilities are put on the stack (see rule 603, "Handling Triggered Abilities").
				 * These steps repeat in order until no further state-based actions are performed and no abilities trigger.
				 * Then the player who would have received priority does so. */
				/* Rule 117.1: ... The player with priority may cast spells, activate abilities, and take special actions. */
				/* Rule 117.1a: A player may cast an instant spell any time they have priority...
				 * Rule 117.1b: A player may activate an activated ability any time they have priority. */
				do {
					do {
						this.performs_all_applicable_state_based_actions_as_a_single_event();
					} while (this.Has_Performed_A_State_Based_Action);
					this.Stack.adds_all_triggered_abilities_in_list_of_triggered_abilities_to_be_added_to_this_to_this();
				} while (this.Has_Performed_A_State_Based_Action || !this.Stack.list_of_triggered_abilities_to_be_added_to_this().isEmpty());
				do {
					this.casts_a_spell_activates_an_activated_ability_or_takes_a_special_action("This Player's " + The_Step_To_Use, this + " begins " + this + "'s " + The_Step_To_Use);
				} while (!this.Has_Passed);
				do {
					do {
						this.performs_all_applicable_state_based_actions_as_a_single_event();
					} while (this.Has_Performed_A_State_Based_Action);
					this.Stack.adds_all_triggered_abilities_in_list_of_triggered_abilities_to_be_added_to_this_to_this();
				} while (this.Has_Performed_A_State_Based_Action || !this.Stack.list_of_triggered_abilities_to_be_added_to_this().isEmpty());
				do {
					this.Other_Player.casts_a_spell_activates_an_activated_ability_or_takes_a_special_action("The Other Player's " + The_Step_To_Use, this + " begins " + this + "'s " + The_Step_To_Use + " and acts");
				} while (!this.Other_Player.Has_Passed);
			} while (this.Other_Player.Has_Taken_An_Action);
			this.Stack.resolves_top_object();
		} while (this.Stack.has_resolved());
	}
	
	/* Rule 503.1: The upkeep step has no turn-based actions.
	 * Once it begins, the active player gets priority. (See rule 117, "Timing and Priority.") */
	public void completes_her_upkeep_step() throws Exception {
		System.out.println(this + " is completing " + this + "'s Upkeep Step.");
		this.performs_state_based_actions_adds_trigged_abilities_to_stack_and_has_this_and_other_player_cast_spells_activate_abiltiies_and_take_special_actions("Upkeep Step");
	}
	
	public a_part_of_the_battlefield part_of_the_battlefield() {
		return this.Part_Of_The_Battlefield;
	}
	
	public ArrayList<a_permanent> determines_her_permanents_to_untap() {
		ArrayList<a_permanent> The_List_Of_Permanents_That_Should_Be_Untapped = new ArrayList<a_permanent>();
		for (an_artifact The_Artifact : this.Part_Of_The_Battlefield.list_of_artifacts()) {
			if (The_Artifact.is_tapped()) {
				The_List_Of_Permanents_That_Should_Be_Untapped.add(The_Artifact);
			}
		}
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			if (The_Creature.is_tapped()) {
				The_List_Of_Permanents_That_Should_Be_Untapped.add(The_Creature);
			}
		}
		for (an_enchantment The_Enchantment : this.Part_Of_The_Battlefield.list_of_enchantments()) {
			if (The_Enchantment.is_tapped()) {
				The_List_Of_Permanents_That_Should_Be_Untapped.add(The_Enchantment);
			}
		}
		for (a_land The_Land : this.Part_Of_The_Battlefield.list_of_lands()) {
			if (The_Land.is_tapped()) {
				The_List_Of_Permanents_That_Should_Be_Untapped.add(The_Land);
			}
		}
		for (a_planeswalker The_Planeswalker : this.Part_Of_The_Battlefield.list_of_planeswalkers()) {
			if (The_Planeswalker.is_tapped()) {
				The_List_Of_Permanents_That_Should_Be_Untapped.add(The_Planeswalker);
			}
		}
		System.out.println(this.Name + " determines the following list of permanents to untap. " + The_List_Of_Permanents_That_Should_Be_Untapped);
		return The_List_Of_Permanents_That_Should_Be_Untapped;
	}
	
	
	public void determines_whether_are_playable_the_cards_in(ArrayList<a_nonland_card> The_List_Of_Nonland_Hand_Cards, String The_Step_To_Use) {
		for (a_nonland_card The_Nonland_Card : The_List_Of_Nonland_Hand_Cards) {
			if (this.indicates_whether_is_playable(The_Nonland_Card, The_Step_To_Use)) {
				The_Nonland_Card.becomes_playable();
			} else {
				The_Nonland_Card.becomes_not_playable();
			}
		}
	}
	
	
	public void determines_whether_her_permanents_nonmana_activated_abilities_are_activatable() throws Exception {
		for (a_permanent The_Permanent : this.Part_Of_The_Battlefield.list_of_permanents()) {
			for (a_nonmana_activated_ability The_Nonmana_Activated_Ability : The_Permanent.list_of_nonmana_activated_abilities()) {
				if (The_Nonmana_Activated_Ability.requires_tapping()) {
					if (The_Permanent.is_tapped()) {
						The_Nonmana_Activated_Ability.becomes_nonactivatable();
					} else {
						if (The_Nonmana_Activated_Ability.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this().size() > 0) {
							The_Nonmana_Activated_Ability.becomes_activatable();
						} else {
							The_Nonmana_Activated_Ability.becomes_nonactivatable();
						}
					}
				} else {
					if (The_Nonmana_Activated_Ability.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this().size() > 0) {
						The_Nonmana_Activated_Ability.becomes_activatable();
					} else {
						The_Nonmana_Activated_Ability.becomes_nonactivatable();
					}
				}
			}
		}
	}
	
	
	public void draws() {
		this.Hand.receives(this.Deck.removes_and_provides_its_top_card());		
		System.out.println("After drawing, " + this + "'s deck has " + this.Deck.number_of_cards() + " cards and contains the following. " + this.Deck);
		System.out.println("After drawing, " + this + "'s hand has " + this.Hand.number_of_cards() + " cards and contains the following. " + this.Hand);
	}
	
	
	/**
	 * draws_a_hand
	 * 
	 * Rule 103.4: Each player draws a number of cards equal to their starting hand size, which is normally seven.
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
    
    
    private ArrayList<a_nonland_card> generates_a_list_of_playable_nonland_hand_cards_in(ArrayList<a_nonland_card> The_List_Of_Nonland_Hand_Cards) {
		ArrayList<a_nonland_card> The_List_Of_Playable_Nonland_Hand_Cards = new ArrayList<a_nonland_card>();
		for (a_nonland_card The_Nonland_Card : The_List_Of_Nonland_Hand_Cards) {
			if (The_Nonland_Card.is_playable()) {
				The_List_Of_Playable_Nonland_Hand_Cards.add(The_Nonland_Card);
			}
		}
		return The_List_Of_Playable_Nonland_Hand_Cards;
    }
    
    
    private ArrayList<a_nonmana_activated_ability> generates_a_list_of_activatable_nonmana_activated_abilities() {
    	ArrayList<a_nonmana_activated_ability> The_List_Of_Activatable_Nonmana_Activated_Abilities = new ArrayList<a_nonmana_activated_ability>();
    	for (a_permanent The_Permanent : this.Part_Of_The_Battlefield.list_of_permanents()) {
    		for (a_nonmana_activated_ability The_Nonmana_Activated_Ability : The_Permanent.list_of_nonmana_activated_abilities()) {
    			if (The_Nonmana_Activated_Ability.is_activatable()) {
    				The_List_Of_Activatable_Nonmana_Activated_Abilities.add(The_Nonmana_Activated_Ability);
    			}
    		}
    	}
    	return The_List_Of_Activatable_Nonmana_Activated_Abilities;
    }
    
    
    private <T> void helps_generate_a_list_of_combinations(ArrayList<T> list_of_elements, int starting_index_in_list_of_elements, ArrayList<T> combination, ArrayList<ArrayList<T>> list_of_combinations) {
        list_of_combinations.add(new ArrayList<>(combination));
        for (int i = starting_index_in_list_of_elements; i < list_of_elements.size(); i++) {
            combination.add(list_of_elements.get(i));
            helps_generate_a_list_of_combinations(list_of_elements, i + 1, combination, list_of_combinations);
            combination.remove(combination.size() - 1);
        }
    }
    
	
	public boolean indicates_whether_is_playable(a_nonland_card The_Nonland_Card, String The_Step_To_Use) {
		if (
			this.indicates_whether_a_card_is_playable_according_to_the_text_of(The_Nonland_Card, The_Step_To_Use) &&
			!The_Nonland_Card.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this().isEmpty()
		) {
			return true;
		} else {
			return false;
		}
	}
	
	
	public boolean indicates_whether_a_card_is_playable_according_to_the_text_of(a_card The_Card, String The_Step_To_Use) {
		if (The_Card.type().equals("Instant")) {
			ArrayList<String> The_Text = The_Card.text();
			ArrayList<a_creature> The_List_Of_Creatures = this.Part_Of_The_Battlefield.list_of_creatures();
			
			for (String The_Line : The_Text) {
				if (The_Line.contains("creature you control") && The_List_Of_Creatures.isEmpty()) {
					return false;
				}
			}
			
			// Tactical Advantage: "Target blocking or blocked creature you control gets +2/+2 until end of turn."
			if (The_Text.contains("Target blocking or blocked creature you control")) {
				if (!The_Step_To_Use.equals("This Player's Declare Blockers Step") && !The_Step_To_Use.equals("Other Player's Declare Blockers Step")) {
					return false;
				} else {
					for (a_creature The_Creature : The_List_Of_Creatures) {
						if (The_Creature.is_blocked()) {
							return true;
						} else if (The_Creature.is_blocking()) {
							return true;
						}
					}
					return false;
				}
			}
			
			else {
				// Stony Strength: "Put a +1/+1 counter on target creature you control. Untap that creature."
				return true;
			}
			
		} else {
		    return true;
		}
	}
	
	
	public void plays_a_land() throws Exception {
		ArrayList<a_land_card> The_List_Of_Land_Cards = this.Hand.list_of_land_cards();
		int The_Number_Of_Land_Cards = The_List_Of_Land_Cards.size();
		System.out.println(this.Name + " is taking the special action of playing a land.");
		int The_Index_Of_The_Land_Card_To_Play = this.Random_Data_Generator.nextInt(0, The_Number_Of_Land_Cards - 1);
		a_land_card The_Land_Card_To_Play = The_List_Of_Land_Cards.get(The_Index_Of_The_Land_Card_To_Play);
		String The_Name_Of_The_Land_Card = The_Land_Card_To_Play.name();
		a_land The_Land = null;
		if (The_Name_Of_The_Land_Card.equals("Plains")) {
			The_Land = new a_land("Plains", this);
			a_mana_ability The_Mana_Ability = new a_mana_ability("T", "Add [W].", The_Land);
			The_Land.receives(The_Mana_Ability);
		} else if (The_Name_Of_The_Land_Card.equals("Forest")) {
			The_Land = new a_land("Forest", this);
			a_mana_ability The_Mana_Ability = new a_mana_ability("T", "Add [G].", The_Land);
			The_Land.receives(The_Mana_Ability);
		} else {
			throw new Exception("The MTG Game Simulator does not know how to play the land with name " + The_Name_Of_The_Land_Card);
		}
		this.Hand.removes(The_Land_Card_To_Play);
		this.Part_Of_The_Battlefield.receives(The_Land);
		this.Has_Passed = false;
		this.Has_Taken_An_Action = true;
		this.Has_Played_A_Land_This_Turn = true;
		System.out.println("After playing a land card, " + this + "'s hand has " + this.Hand.number_of_cards() + " cards and contains the following. " + this.Hand);
		System.out.println("After playing a land card, " + this + "'s part of the battlefield has " + this.Part_Of_The_Battlefield.number_of_permanents() + " cards and contains the following. " + this.Part_Of_The_Battlefield);
	}
	
	
	public String name() {
		return this.Name;
	}
	
	
	/**
	 * receives_priority_and_acts
	 * 
	 * Rule 117.1: Unless a spell or ability is instructing a player to take an action, which player can take actions at any given time is determined by a system of priority...
	 */
	public void casts_a_spell_activates_an_activated_ability_or_takes_a_special_action(String The_Step_To_Use, String The_Event_After_Which_She_Is_Receiving_Priority_And_Acting) throws Exception {
		System.out.println(this + " is considering casting a spell, activating an ability, or taking a special action.");
		if (this.Stack.isEmpty()) {
			if (The_Step_To_Use.equals("This Player's Main Phase")) {
				/* Rule 116.2a: Playing a land is a special action.
				 * To play a land, a player puts that land onto the battlefield from the zone it was in (usually that player's hand).
				 * By default, a player can take this action only once during each of their turns.
				 * A player can take this action any time they have priority and the stack is empty during a main phase of their turn. See [R]ule 305, "Lands."
				 * Rule 505.5b: During either main phase, the active player may play one land card from their hand if the stack is empty, if the player has priority, and if they haven't played a land this turn (unless an effect states the player may play additional lands).
				 * This action doesn't use the stack.
				 * Neither the land nor the action of playing the land is a spell or ability, so it can't be countered, and players can't respond to it with instants or activated abilities.
				 * (See rule 305, "Lands.")
				 */
				if (!this.Has_Played_A_Land_This_Turn && !this.Hand.list_of_land_cards().isEmpty() /*&& an_enumeration_of_states_of_a_coin.provides_a_state() == an_enumeration_of_states_of_a_coin.HEADS*/) {
					this.plays_a_land();
				}
				/* Rule 117.1a: ... A player may cast a non-instant spell during their main phase any time they have priority and the stack is empty. */
				else {
					this.casts_a_spell_or_activates_a_nonmana_activated_ability(The_Step_To_Use, false);
				}
			/* Rule 117.7: If a player with priority casts a spell or activates an activated ability while another spell or ability is already on the stack, the new spell or ability has been cast or activated "in response to" the earlier spell or ability.
			 * The new spell or ability will resolve first. See rule 608, "Resolving Spells and Abilities." */
			} else {
				this.casts_a_spell_or_activates_a_nonmana_activated_ability(The_Step_To_Use, true);
			}
		}
		else {
			this.casts_a_spell_or_activates_a_nonmana_activated_ability(The_Step_To_Use, true);
		}
	}
	
	
	public void receives(a_player The_Other_Player_To_Use) {
		this.Other_Player = The_Other_Player_To_Use;
	}
	
	
	public void shuffles_her_deck() {
		this.Deck.shuffles();
		System.out.println(
			"The deck of " + this.Name + " after shuffling has " + this.Deck.number_of_cards() + " cards and is the following. " + this.Deck
		);
	}
	
	public void takes_her_turn() throws Exception {
		System.out.println(this.Name + " is taking their turn.");
		/* Rule 117.3a: The active player receives priority at the beginning of most steps and phases, after any turn-based actions (such as drawing a card during the draw step; see rule 703) have been dealt with and abilities that trigger at the beginning of that phase or step have been put on the stack.
		 * Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		 * Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.") */
		this.completes_her_beginning_phase();
		this.completes_a_main_phase();
		this.completes_her_combat_phase();
		this.completes_a_main_phase();
		this.completes_her_end_phase();
		
		// Rule 500.5: Effects that last "until end of turn" are subject to special rules; see rule 514.2.	
	}
	
	
	public void untaps_her_permanents(ArrayList<a_permanent> The_List_Of_Permanents_That_Should_Be_Untapped) {
		System.out.println(this.Name + " is untapping their permanents.");
		for (a_permanent The_Permanent : The_List_Of_Permanents_That_Should_Be_Untapped) {
			The_Permanent.untaps();
		}
	}
	
	public a_player other_player() {
		return this.Other_Player;
	}
	
	
	@Override
	public String toString() {
		return this.Name;
	}

	
}