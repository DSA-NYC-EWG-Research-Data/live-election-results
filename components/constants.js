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
  'Eric L. Adams': '#8f49b7',
  'Irene Estrada': '#a6cee3',
  'Joseph Hernandez': '#fc9a98',
  'Andrew M. Cuomo': '#1660a8',
  'Curtis A. Sliwa': '#ef4b30',
  'Zohran Kwame Mamdani': '#ffab00',
  'Jim Walden': '#6bb23d',
  'Alexa Avilés': 'rgb(199,72,82)',
  'Luis E. Quero': 'rgb(58,59,115)',
  'WRITE-IN': '#a5a5a5',
}

export const candidates = {
  Mayoral: [
    'Eric L. Adams',
    'Curtis A. Sliwa',
    'Andrew M. Cuomo',
    'Irene Estrada',
    'Joseph Hernandez',
    'Zohran Kwame Mamdani',
    'Jim Walden',
  ],
  'City Council 38': ['Alexa Avilés', 'Luis E. Quero'],
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
