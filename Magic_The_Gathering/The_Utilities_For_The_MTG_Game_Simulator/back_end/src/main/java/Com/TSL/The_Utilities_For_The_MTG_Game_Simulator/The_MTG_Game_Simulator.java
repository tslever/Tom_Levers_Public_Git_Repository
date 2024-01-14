package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class The_MTG_Game_Simulator {
	
    public static void main(String[] args) throws Exception {
    	a_stack The_Stack = new a_stack();
    	a_player The_First_Player = new a_player("Tom", The_Stack);
    	a_deck_builder The_Deck_Builder = new a_deck_builder();
    	a_deck The_Deck_Keep_The_Peace = The_Deck_Builder.builds_Keep_The_Peace();
    	//a_deck_history The_Deck_History_For_Keep_the_Peace = new a_deck_history(The_Deck_Keep_The_Peace, 0, 0);
    	//System.out.println(The_Deck_History_For_Keep_the_Peace);
    	The_First_Player.receives(The_Deck_Keep_The_Peace);
    	a_player The_Second_Player = new a_player("Scott", The_Stack);
    	a_deck The_Deck_Large_And_In_Charge = The_Deck_Builder.builds_Large_And_In_Charge();
    	//a_deck_history The_Deck_History_For_Large_and_in_Charge = new a_deck_history(The_Deck_Large_And_In_Charge, 0, 0);
    	//System.out.println(The_Deck_History_For_Large_and_in_Charge);
    	The_Second_Player.receives(The_Deck_Large_And_In_Charge);
    	The_First_Player.receives(The_Second_Player);
    	The_Second_Player.receives(The_First_Player);
    	a_game The_Game = new a_game(The_First_Player, The_Second_Player);
    	The_Game.plays();
    }
}