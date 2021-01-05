
// Allows executable to find class Main.
package com.mycompany.serverutilities;

// Imports classes.
import com.mycompany.serverutilities.clientcommunicationutilities.Server;
import com.mycompany.serverutilities.clientcommunicationutilities.IceCreamClientCommunication;
import com.mycompany.serverutilities.productutilities.IceCreamProductRetrieval;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import org.apache.commons.lang.StringUtils;

/**
 * Defines class Main that encapsulates static method main, the entry point of
 * this application.
 * @version 0.0
 * @author Tom Lever
 */
public class Main {
    
    /**
     * Defines method main of this application, which instantiates a server,
     * an ice cream product retrieval subsystem, and an ice cream client
     * communication subsystem based on the server and the retrieval subsystem.
     * 
     * @param args
     * @throws Exception
     * @throws IOException
     */
    public static void main(String[] args) throws Exception, IOException {
        System.out.println("main: Started Tom's ice-cream server program.");
        
        // isNumeric returns true if argument is a non-negative integer.
        if ((args.length != 1) || (!StringUtils.isNumeric(args[0]))) {
            throw new Exception(
                "main: First command-line argument has been found to not " +
                "exist or not be a non-negative integer.");
        }
        // TODO: Generate InvalidPortNumberException when args[0] is found to
        // not exist or to not be a non-negative integer.
        
        int portNumber = Integer.parseInt(args[0]);
        System.out.printf(
            "main: Parsed first command-line argument as port number %d.\n",
            portNumber);
        
        Server server = new Server(
            HttpServer.create(new InetSocketAddress(portNumber), 0));
        // TODO: Generate CreateException when IOException is thrown by create.
        System.out.println(
            "main: Created server, an instance of Server, that will listen " +
            "on IP address : port 'localhost : " + portNumber + "' for HTTP " +
            "messages.");    
        
        IceCreamProductRetrieval retriever = new IceCreamProductRetrieval();
        System.out.println(
            "main: Created retriever, an instance of " +
            "IceCreamProductRetrieval, that represents the server's ice " +
            "cream product retrieval subsystem.");
        
        IceCreamClientCommunication communicator =
            new IceCreamClientCommunication(server, retriever);
        System.out.println(
            "main: Created communicator, an instance of " +
            "IceCreamClientCommunication, that represents the server's ice " +
            "cream client communication subsystem.");
        
        communicator.setMessageInterfacesAndStartServerListening();
        System.out.println(
            "main: Set message interfaces and started server listening.");
    }
}
