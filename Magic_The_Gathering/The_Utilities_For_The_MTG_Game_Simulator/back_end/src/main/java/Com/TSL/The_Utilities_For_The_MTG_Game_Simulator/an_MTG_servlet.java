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

	a_game Game;
	Gson Gson;
	
	public an_MTG_servlet() {
		this.Game = new a_game();
		this.Gson = new Gson();
	}
	
    /**
     * Responds to a GET request by advancing this MTG servlet's game.
     * 
     * @param request a GET request
     * @param response a response to a GET request
     * @throws IOException 
     */
    @Override
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        response.setHeader("Access-Control-Allow-Origin", "http://localhost:3000");
        String The_Summary_Of_The_Next_Action = this.Game.advances();
        try ( PrintWriter printWriter = response.getWriter(); ) {
        	printWriter.println(this.Gson.toJson(The_Summary_Of_The_Next_Action));
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
