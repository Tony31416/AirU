#include <Adafruit_NeoPixel.h> //include the adafruit library needed to control the neopixels
#ifdef __AVR__
  #include <avr/power.h>  //include power use reduction code, if it's not already there
#endif

// define constants
#define NEOPIN      6   //digital pin where the neopixel is plugged in
#define NEONUM      8   //number of neopixel leds
#define SENPIN      A5  //the pin for the analog in from the photoresistor
#define SENPOW      3   //the 5V power for the sensor
#define SENGND      2   //the ground for the sensor
#define AVGNUM     100  //the number of measurements to average

// define variables
int     senval=0;         //the reading from the photoresistor
int     minsenval=2000; //the minimum reading
int     maxsenval=0;    //the minimum reading
int     percent=0;      //the percent between the min and max
int     nlights=0;      //the number of lights to turn on
int     iavg=0;         //counter for average
int     avgsum=0;
float   meansens=0;     //average sensor value
float   tau=200;        //time constant for low pass filter
float   alpha=10;       //smoothing factor, high alpha leads to less noise but also less sensitivity
float   filtsenval=0;   //filtered sensor value
int     dt=0;           //change in time from one datum to another
int     t=0;            //time
int     i=0;            //counter for loops

// Initialize the 'neopixel' object, which will be the object used to communicate with the
// indicator lights, by sending each one a red green or blue value
Adafruit_NeoPixel neopixels = Adafruit_NeoPixel(NEONUM, NEOPIN, NEO_GRB + NEO_KHZ800);

void setup() {        //initialization code, only runs at the board's power up
  Serial.begin(9600); //start talking to a computer if connected
  
  pinMode(SENPOW, OUTPUT);      // sets this digital pin as output
  pinMode(SENGND, OUTPUT);      // sets this digital pin as output
  digitalWrite(SENPOW, HIGH);   // sets the sensor's circuit power to 5V
  digitalWrite(SENGND, LOW);    // sets the sensor's ground to 0V
  
  neopixels.begin(); // This initializes the NeoPixel software library.
  for(int i=0;i<NEONUM;i++){ //for all neopixel lights, set to an initial blue color
    neopixels.setPixelColor(i, neopixels.Color(1,1,50)); //set color to rgb value (blue here)
  }
  neopixels.show();  //turn on the lights to the color they were set to above
  delay(500);  //wait half a second (500 milliseconds) to show the user that the board reset
  meansens=analogRead(SENPIN);    //average sensor value

  //In this loop for three seconds, find the average low light reading from the sensor
  //Make sure no particulate matter is going through the sensor at this point
  while (millis()<3000){
    i++;  //counter
    senval=analogRead(SENPIN);   //read from the sensor
    meansens=(meansens*(i)+senval)/(i+1);  //calculate a new mean
    if (minsenval>senval) {minsenval=senval;}  //find a new possible min
    if (maxsenval<senval) {maxsenval=senval;}  //find a new possible max
    delay(50);   //pause for 0.05 seconds
    Serial.print("Calibrating, meas: ");    //write results to a computer, if attached
    Serial.print(senval); Serial.print("  Mean: ");
    Serial.println(meansens);
  }
  for(int i=0;i<NEONUM;i++){ //for all neopixel lights, set to an initial cyan color
    neopixels.setPixelColor(i, neopixels.Color(1,51,50)); //set color to rgb value (cyan here)
  }
  neopixels.show();  //turn on the lights to the color they were set to above
  delay(500);  //wait half a second (500 milliseconds) to show the user that the board calibrated
  
  int range=maxsenval-maxsenval;    //calculate the range in the low light condition
  minsenval=meansens;               //minimum sensor value
  maxsenval=maxsenval+range;        //maximum sensor value

  pinMode(13, OUTPUT);      // digital pin linked to the led indicator on board
  digitalWrite(13, HIGH);   // turn on the led indicator to again show calibration is over
}

void loop() {     //the code in this loop runs repeatedly, as long as the board is powered
  delay(50);      //pause 0.05 seconds
  dt=millis()-t;  //change in time since the last reading
  t=millis();     //current time in milliseconds
  alpha=dt/(tau+dt);          //filter value
  senval=analogRead(SENPIN);  //read from the sensor
  
  filtsenval=filtsenval+alpha*(senval-filtsenval); //the filtered sensor reading
  
  iavg++;                     //counter for moving window average
  if (iavg>=AVGNUM) iavg=0;   //reset iavg each time the max is reached
  if (maxsenval<filtsenval) {maxsenval=filtsenval;}   //keep lookinmg for and updated the max sensor reading
  percent=round(100*(filtsenval-minsenval)/(maxsenval-minsenval));  //calculte the percent the sensor reading is between its max and min
  nlights=round(percent*NEONUM/100);      //turn on a number of lights corresponding to the percent sensor reading (100% = all on)
  if (nlights<1) {nlights=1; percent=0;}  //keep at least one light always on, don't allow a percentage < 0%
  if (percent>100) {nlights=NEONUM; percent=100;}  //don't allow a percentage above 100%

  //Print data to computer, if attached
  Serial.print(senval);     Serial.print("__"); Serial.print(minsenval);  Serial.print("__");  
  Serial.print(maxsenval);  Serial.print("__"); Serial.print(nlights);    Serial.print("__");
  Serial.print(percent);    Serial.print("__"); Serial.println(filtsenval);
  
  for(int i=0;i<nlights;i++){         //loop through the lights and turn them on as needed
    neopixels.setPixelColor(i, neopixels.Color(percent,100-percent,0));  //turn these on to a grb value between green and red depending upon percent
  }
  neopixels.show(); // This sends the updated pixel color to the hardware.
  for(int i=nlights;i<NEONUM;i++){    //turn off other lights
    neopixels.setPixelColor(i, neopixels.Color(0,0,0)); 
  }
  neopixels.show(); // This sends the updated pixel color to the hardware.
}
