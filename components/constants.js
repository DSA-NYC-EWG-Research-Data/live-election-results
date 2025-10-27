export const boroughColors = {
  'The Bronx': '#77a9b8',
  Queens: '#f5d578',
  Manhattan: '#82c3f5',
  'Staten Island': '#f7746d',
  Brooklyn: '#f29357',
}

export const boroughColorsDarker = {
  'The Bronx': '#227a94',
  Queens: '#b88f18',
  Manhattan: '#2480c7',
  'Staten Island': '#bf2119',
  Brooklyn: '#ba5718',
}

export const candidateColors = {
  'Eric L Adams': '#8f49b7',
  'Andrew Cuomo': '#1660a8',
  'Irene Estrada': '#a6cee3',
  'Joseph Hernandez': '#6bb23d',
  'Zohran Kwame Mamdani': '#ffab00',
  'Curtis Sliwa': '#ef4b30',
  'Jim Walden': '#fc9a98',
  'Alexa Avilés': 'rgb(199,72,82)',
  'Republican Candidate': '#cc6633',
  'Write In': '#a5a5a5',
}

export const candidates = {
  Mayoral: [
    'Eric L Adams',
    'Andrew Cuomo',
    'Irene Estrada',
    'Joseph Hernandez',
    'Zohran Kwame Mamdani',
    'Curtis Sliwa',
    'Jim Walden',
  ],
  'City Council 38': ['Alexa Avilés', 'Republican Candidate'],
}

export const raceLookup = {
  Mayoral: 'mayoral',
  'City Council 38': 'council',
}

const maptilerKey = process.env.NEXT_PUBLIC_MAPTILER_KEY

export const mapStyles = {
  color: `https://api.maptiler.com/maps/01968205-0dc7-71df-87a7-8b67f7828379/style.json?key=${maptilerKey}`,
  monochrome: `https://api.maptiler.com/maps/01961350-1791-703e-8753-2c795c604620/style.json?key=${maptilerKey}`,
}

export const scaleLookup = {
  'Assembly district': 'assembly-district',
  'Election district': 'election-district',
}
