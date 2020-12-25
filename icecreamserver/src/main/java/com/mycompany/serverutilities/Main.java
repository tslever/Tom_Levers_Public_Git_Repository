
// Allows executable to find class Main.
package com.mycompany.serverutilities;

// Import classes.
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import org.apache.commons.lang.StringUtils;

/**
 * Defines Main class that encapsulates static method main,
 * the entry point of this application.
 * This application instantiates a Server based on an httpServer object,
 * calls Server's setMessageInterfaces.
 * @version 0.0
 * @author Tom Lever
 */
public class Main {
    
    /**
     * Defines the main method of this application.
     * @param args
     * @throws IOException, Exception
     */
    public static void main(String[] args) throws Exception, IOException {
        
        System.out.println("Starting Tom's ice cream server program.");
        
        
        Config config = new Config();
        
        
        // ---------------------
        // Construct the server.
        // ---------------------
        // Check that portNumber is in command-line arguments and is numeric.
        if ((args.length != 1) || (!StringUtils.isNumeric(args[0]))) {
            throw new Exception(
                "Usage error. Usage: java EchoServer <portNumber>");
        }
        
        // Parse portNumber from command-line arguments.
        int portNumber = Integer.parseInt(args[0]);
        System.out.printf("User-specified Port number: %d.\n", portNumber);
        
        /*
         * Create an httpServer object that listens on localhost:<portNumber>.
         * localhost specifies this computer (company) and port <portNumber>
         * identifies the port (desk) that will handle communications.
         * The httpServer is like a person who sits at this desk who will
         * answer the phone and respond to each HTTP request. Formally,
         * an HttpServer is a class that decodes TCP/UDP messages into HTTP
         * messages; parses incoming HTTP messages; performs functionality like
         * CRUD operations; and provides responses to clients.
         */
        Server server = new Server(
            HttpServer.create(new InetSocketAddress(portNumber), 0));
        // TODO: Generate CreateException when IOException is thrown by create.
        

        server.setMessageInterfaces(config.getMessageInterfaces());
        // TODO: Handle exceptions from setMessageInterfaces and
        // startListeningForMessages.
        
        
        // ------------------------------------------------------------------
        /* Have server listen for messages from clients and provide responses
         * to clients. */
        // ------------------------------------------------------------------
        server.startListeningForMessages();
    }
}
