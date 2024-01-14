package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;

import org.apache.commons.io.IOUtils;

import com.google.gson.Gson;

import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * an_echo_servlet is a servlet of an HTTP / Jetty / Gretty server that echoes a request in the response body.
 */
@WebServlet("/echo")
public class an_MTG_servlet extends HttpServlet {

	a_game The_Game;
	
	public an_MTG_servlet() {
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
	}
	
    /**
     * Responds to a GET request by echoing a request in the response body.
     * 
     * @param request a GET request
     * @param response a response to a GET request
     */
    @Override
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        response.setHeader("Access-Control-Allow-Origin", "http://localhost:3000");
        String firstLine = request.getMethod() + " " + request.getRequestURI();
        Gson gson = new Gson();
        String The_Test_JSON = gson.toJson("test string");
        try (
            PrintWriter printWriter = response.getWriter();
        ) {
        	printWriter.println(The_Test_JSON);
        }
    }
    
    /**
     * Responds to a POST request by echoing a request in the response body.
     * 
     * @param request A POST request
     * @param response a response to a POST request
     */
    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        doGet(request, response);
    }
}
