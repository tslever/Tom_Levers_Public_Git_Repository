package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;

import org.apache.commons.io.IOUtils;

import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * an_echo_servlet is a servlet of an HTTP / Jetty / Gretty server that echoes a request in the response body.
 */
@WebServlet("/echo")
public class an_MTG_servlet extends HttpServlet {

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
        if (request.getQueryString() != null) {
            firstLine += "?" + request.getQueryString();
        }
        firstLine += " " + request.getProtocol();
        try (
            PrintWriter printWriter = response.getWriter();
            BufferedReader bufferedReader = request.getReader();
        ) {
            printWriter.println(firstLine);
            for (String headerName : Collections.list(request.getHeaderNames())) {
                for (String headerValue : Collections.list(request.getHeaders(headerName))) {
                    printWriter.println(headerName + ": " + headerValue);
                }
            }
            printWriter.println();
            IOUtils.copy(bufferedReader, printWriter);
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
