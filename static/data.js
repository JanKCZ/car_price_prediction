var fuell_data = ['benzín', 'nafta', 'hybridní', 'LPG + benzín', 'CNG + benzín', 'ethanol', 
       'elektro', 'jiné'];

var transmission_data = ['manuální', "poloautomatická", "automatická"];

var air_condition_data = ['manuální', 'automatická', 'dvouzónová automatická',
	       'třízónová automatická', 'čtyřzónová automatická',
	       'bez klimatizace'];

var country_data = ['Česká republika', 'Německo', 'Belgie', 'Švýcarsko',
	       'Itálie', 'Slovenská republika', 'Holandsko', 'Rakousko',
	       'Francie', 'Lucembursko', 'USA', 'Dánsko', 'Španělsko', 'Jiná',
	       'Nedohledatelný původ'];

var car_type_data = ['hatchback', 'sedan/limuzína', 'MPV', 'liftback', 'SUV', 'kombi',
	       'kupé', 'VAN', 'pick-up', 'kabriolet', 'terénní', 'CUV',
	       'roadster'];

var car_brand_model_dict = {'Abarth': ['595', '500', '695', 'Punto'],
 'Acura': ['MDX'],
 'Aixam': ['CROSSOVER', 'Minauto', 'GTO', 'CITY', 'CROSSLINE', 'Coupé'],
 'Alfa Romeo': ['159',
  '147',
  'Giulietta',
  'Giulia',
  'Stelvio',
  '156',
  'MiTo',
  'GT',
  'Brera',
  '166',
  'GTV',
  'Crosswagon Q4',
  'Spider',
  '146',
  '164',
  '145'],
 'Alpina': ['B7', 'D3', 'B3 S', 'B5', 'D5'],
 'Aston Martin': ['DB9', 'Vanquish', 'V8 Vantage'],
 'Audi': ['A4',
  'A6',
  'A3',
  'Q7',
  'A5',
  'Q5',
  'A6 allroad',
  'A8',
  'Q3',
  'A6 Avant',
  'A7',
  'A4 Avant',
  'A4 allroad',
  'TT',
  'SQ7',
  'RS6',
  'Q8',
  'A1',
  'S6',
  'S5',
  'SQ5',
  'Q2',
  'S3',
  'SQ8',
  'A2',
  'S8',
  'R8',
  'S4',
  'RS7',
  'S7',
  'RS5',
  'RS3',
  'RS4',
  'RS Q8',
  'e-tron',
  '80',
  '200',
  'RS4 Avant',
  'RS2',
  'Coupé',
  'TTS'],
 'BMW': ['Řada 3',
  'Řada 5',
  'X5',
  'X3',
  'Řada 1',
  'X1',
  'X6',
  'Řada 7',
  'Řada 4',
  'Řada 2',
  'Řada 6',
  'X4',
  'X7',
  'Z4',
  'M4',
  'Řada 8',
  'X2',
  'M5',
  'M3',
  'i3',
  'M2',
  'M6',
  'i8',
  'Z3'],
 'Bentley': ['Continental GT',
  'Bentayga',
  'Continental GTC',
  'Continental Flying Spur',
  'Arnage',
  'Brooklands'],
 'Buick': ['Enclave'],
 'Cadillac': ['Escalade', 'SRX', 'XT5', 'CTS', 'BLS', 'Seville', 'ATS'],
 'Chevrolet': ['Captiva',
  'Aveo',
  'Cruze',
  'Orlando',
  'Spark',
  'Camaro',
  'Lacetti',
  'Corvette',
  'Matiz',
  'Kalos',
  'Nubira',
  'Trax',
  'Tacuma',
  'Silverado',
  'Tahoe',
  'Epica',
  'Suburban',
  'Blazer',
  'Trans Sport',
  'Rezzo',
  'Equinox',
  'HHR',
  'Avalanche'],
 'Chrysler': ['Grand Voyager',
  'Pacifica',
  'PT Cruiser',
  'Voyager',
  'Town & Country',
  '300C',
  'Sebring',
  'Crossfire',
  '300M',
  'ES'],
 'Citroën': ['Berlingo',
  'C3',
  'C4 Picasso',
  'C5',
  'C4',
  'Grand C4 Picasso',
  'Xsara Picasso',
  'C3 Picasso',
  'C4 Cactus',
  'C3 Aircross',
  'C1',
  'C5 Aircross',
  'SpaceTourer',
  'C8',
  'C-Crosser',
  'C2',
  'Xsara',
  'C-Elysée',
  'Jumpy',
  'Grand C4 SpaceTourer',
  'DS3',
  'C4 SpaceTourer',
  'C4 Aircross',
  'DS4',
  'DS5',
  'Saxo',
  'C6',
  'Xantia',
  'Jumper',
  'Nemo',
  'BX',
  'Évasion'],
 'Cupra': ['Ateca', 'Leon', 'Formentor'],
 'DS': ['7 Crossback', '3 Crossback', '4', '5', '3'],
 'Dacia': ['Duster', 'Sandero', 'Logan', 'Dokker', 'Lodgy'],
 'Daewoo': ['Matiz', 'Nubira', 'Kalos', 'Lanos', 'Leganza', 'Tacuma'],
 'Daihatsu': ['Terios', 'Sirion', 'Cuore', 'Copen', 'Materia', 'Trevis'],
 'Dodge': ['Ram',
  'Challenger',
  'Durango',
  'Nitro',
  'Caliber',
  'Charger',
  'Grand Caravan',
  'Avenger',
  'Journey',
  'Ram 1500',
  'Stratus'],
 'Ferrari': ['California', '458', 'F12', 'F 430', '360', 'FF', '599', '456'],
 'Fiat': ['Tipo',
  'Punto',
  '500',
  'Panda',
  'Grande Punto',
  'Freemont',
  'Dobló',
  'Stilo',
  'Bravo',
  'Croma',
  'Sedici',
  'Scudo',
  '500X',
  '500L',
  'Ulysse',
  'Ducato',
  'Idea',
  'Qubo',
  'Punto EVO',
  'Talento',
  'Linea',
  'Fiorino',
  '500C',
  'Multipla',
  'Brava',
  'Barchetta',
  '500E',
  'Coupé',
  'Seicento',
  'Marea',
  'Palio',
  '500 Abarth',
  '124 Spider',
  'Fullback',
  'Fiorino Combi'],
 'Ford': ['Focus',
  'Mondeo',
  'S-MAX',
  'Fiesta',
  'C-MAX',
  'Kuga',
  'Galaxy',
  'Fusion',
  'Mustang',
  'Tourneo Custom',
  'Ranger',
  'Grand C-MAX',
  'Transit',
  'Tourneo Connect',
  'Puma',
  'Transit Custom',
  'Ka',
  'Edge',
  'F-150',
  'Tourneo Courier',
  'B-MAX',
  'Ecosport',
  'Transit Connect',
  'Maverick',
  'KA+',
  'Explorer',
  'Streetka',
  'Connect',
  'Escort',
  'Crown Victoria',
  'Expedition',
  'Taurus'],
 'GMC': ['Sierra'],
 'Gonow': ['GS2'],
 'Great Wall': ['Hover H5'],
 'Honda': ['CR-V',
  'Civic',
  'Jazz',
  'HR-V',
  'Accord',
  'FR-V',
  'CR-Z',
  'Insight',
  'Legend',
  'Stream',
  'City',
  'Shuttle',
  'e',
  'Odyssey'],
 'Hummer': ['H2', 'H3', 'H2 SUT', 'H1'],
 'Hyundai': ['i30',
  'Tucson',
  'Santa Fe',
  'i20',
  'ix35',
  'ix20',
  'i40',
  'i10',
  'Getz',
  'Kona',
  'Accent',
  'Matrix',
  'IONIQ',
  'H 1',
  'Elantra',
  'ix55',
  'Terracan',
  'Sonata',
  'Genesis',
  'Trajet',
  'Coupé',
  'Atos',
  'Veloster',
  'Grandeur',
  'Galloper',
  'Lantra',
  'Palisade'],
 'Infiniti': ['FX30',
  'Q50',
  'FX37',
  'QX70',
  'G37',
  'FX50',
  'Q70',
  'QX80',
  'FX35',
  'Q30',
  'EX37',
  'QX56',
  'M30',
  'M37',
  'QX50',
  'QX30',
  'G35'],
 'Isuzu': ['D-Max'],
 'Iveco': ['Massif'],
 'Jaguar': ['XF',
  'F-Pace',
  'XE',
  'E-Pace',
  'F-Type',
  'X-Type',
  'XJ',
  'S-Type',
  'XK',
  'XKR',
  'XJR',
  'XJ8',
  'XK8'],
 'Jeep': ['Grand Cherokee',
  'Compass',
  'Renegade',
  'Wrangler',
  'Cherokee',
  'Commander',
  'Patriot',
  'Gladiator',
  'Wrangler Sahara'],
 'Kia': ['Cee´d',
  'Sportage',
  'Rio',
  'Sorento',
  'Stonic',
  'Picanto',
  'Carens',
  'XCEED',
  'Venga',
  'ProCeed',
  'Niro',
  'Soul',
  'Optima',
  'Carnival',
  'Magentis',
  'Cerato',
  'Stinger',
  'Pro_cee´d',
  'e-Niro',
  'Shuma',
  'e-Soul'],
 'Lada': ['Niva', 'Kalina'],
 'Lamborghini': ['Huracán', 'Gallardo', 'Murciélago'],
 'Lancia': ['Delta',
  'Y',
  'Voyager',
  'Phedra',
  'Lybra',
  'Musa',
  'Thesis',
  'Thema',
  'Kappa'],
 'Land Rover': ['Discovery',
  'Range Rover Evoque',
  'Range Rover',
  'Discovery Sport',
  'Range Rover Sport',
  'Freelander',
  'Defender',
  'Range Rover Velar'],
 'Lexus': ['RX 450h',
  'NX 300h',
  'UX 250h',
  'ES 300h',
  'CT 200h',
  'GS 450h',
  'RX',
  'RX 400h',
  'IS 220d',
  'LC 500',
  'GS 300h',
  'GS',
  'IS 250',
  'IS 300h',
  'LS 500h',
  'ES',
  'IS F',
  'LS 600h',
  'UX 200',
  'RX 300',
  'RX 450h L',
  'GS 300',
  'IS',
  'RX 350',
  'LX 570',
  'IS 200',
  'GS 250',
  'RC 300h',
  'RC',
  'RX 400',
  '300',
  'LS 500',
  'RC F',
  'LX 470',
  'LS 400',
  'SC'],
 'Ligier': ['JS 50', 'JS 50 L'],
 'Lincoln': ['Navigator', 'Town Car', 'Aviator', 'LS'],
 'Lotus': ['Elise'],
 'MG': ['F', 'ZS'],
 'Maserati': ['Ghibli',
  'Levante',
  'Quattroporte',
  'GranTurismo',
  'Coupe',
  'GranCabrio'],
 'Maybach': [],
 'Mazda': ['6',
  '3',
  'CX-5',
  '5',
  'CX-3',
  'CX-30',
  '2',
  'CX-7',
  'MX-5',
  'Tribute',
  'RX-8',
  '323',
  'CX-9',
  'Premacy',
  'BT',
  'MX-30',
  'Demio',
  'CX',
  'MPV',
  '626'],
 'McLaren': ['MP4-12C', '570', '540', '650S Spider'],
 'Mercedes-Benz': ['Třídy E',
  'Třídy C',
  'Třídy A',
  'Třídy S',
  'Třídy B',
  'GLE',
  'GLC',
  'Třídy M',
  'Třídy V',
  'CLA',
  'Vito',
  'CLS',
  'GL',
  'GLS',
  'GLA',
  'Třídy G',
  'Viano',
  'SLK',
  'GLK',
  'Třídy R',
  'SL',
  'AMG GT',
  'CLK',
  'Sprinter',
  'GLB',
  'CL',
  'Třídy X',
  'Citan',
  '124',
  'Vaneo',
  'CLC',
  '220',
  '123',
  '170',
  '190',
  'SLS AMG',
  '200',
  'Maybach',
  'SLC'],
 'Microcar': ['DUE', 'MGO'],
 'Mini': ['Cooper',
  'Countryman',
  'One',
  'Clubman',
  'Cooper S',
  'Paceman',
  'New Mini'],
 'Mitsubishi': ['ASX',
  'Outlander',
  'Eclipse Cross',
  'L200',
  'Colt',
  'Space Star',
  'Lancer',
  'Pajero',
  'Grandis',
  'Carisma',
  'Pajero Pinin',
  'Pajero Sport',
  'SpaceWagon',
  'Eclipse',
  'Galant'],
 'Morgan': [],
 'Nissan': ['Qashqai',
  'Juke',
  'X-Trail',
  'Micra',
  'Navara',
  'Note',
  'Primera',
  'Almera',
  'Pathfinder',
  'Murano',
  'Almera Tino',
  'Patrol',
  'GT-R',
  '350 Z',
  '370 Z',
  'NV200',
  'Tiida',
  'Terrano II',
  'Pixo',
  'LEAF',
  'Terrano',
  'Double cab',
  'Primastar',
  'Maxima',
  'Pulsar',
  'e-NV200',
  'Serena',
  'Evalia'],
 'Opel': ['Astra',
  'Corsa',
  'Zafira',
  'Insignia',
  'Meriva',
  'Crossland X',
  'Combo',
  'Grandland X',
  'Vectra',
  'Vivaro',
  'Mokka',
  'Antara',
  'Agila',
  'Signum',
  'Adam',
  'Tigra',
  'Omega',
  'Cascada',
  'Frontera',
  'Ampera',
  'Speedster'],
 'Peugeot': ['308',
  '5008',
  '207',
  '206',
  '3008',
  '2008',
  '307',
  '508',
  '208',
  'Rifter',
  '407',
  'Partner',
  'Partner Tepee',
  'Traveller',
  '807',
  '301',
  '107',
  '108',
  'Expert Tepee',
  '607',
  '1007',
  '406',
  '4007',
  'Expert',
  'RCZ',
  'Boxer',
  '306',
  '106',
  'Bipper Tepee',
  '4008',
  '405',
  '309'],
 'Porsche': ['Cayenne',
  '911',
  'Panamera',
  'Macan',
  'Cayman',
  '987 Boxster',
  '944',
  '986 Boxster',
  '928'],
 'Renault': ['Mégane',
  'Clio',
  'Scénic',
  'Captur',
  'Grand Scénic',
  'Laguna',
  'Trafic',
  'Espace',
  'Thalia',
  'Kadjar',
  'Modus',
  'Twingo',
  'Kangoo',
  'Fluence',
  'Koleos',
  'Talisman',
  'Grand Espace',
  'Master',
  'Vel Satis',
  'Latitude',
  'ZOE',
  'Alpine',
  'Alaskan',
  'Avantime'],
 'Rolls-Royce': ['Ghost', 'Wraith', 'Phantom'],
 'Rover': ['75', '45', '25'],
 'Saab': ['9-3', '9-5'],
 'Seat': ['Leon',
  'Ibiza',
  'Alhambra',
  'Altea',
  'Ateca',
  'Toledo',
  'Tarraco',
  'Arona',
  'Cordoba',
  'Mii',
  'Exeo',
  'Arosa'],
 'Smart': ['Fortwo', 'Forfour', 'Roadster'],
 'SsangYong': ['Korando', 'Rexton', 'Tivoli', 'Musso', 'Kyron', 'Actyon'],
 'Subaru': ['Outback',
  'Forester',
  'XV',
  'Legacy',
  'Impreza',
  'Justy',
  'Levorg',
  'WRX STI',
  'BRZ',
  'Tribeca',
  'Trezia',
  'SVX'],
 'Suzuki': ['Vitara',
  'Swift',
  'SX4',
  'Ignis',
  'SX4 S-Cross',
  'Grand Vitara',
  'Jimny',
  'S-Cross',
  'Wagon R',
  'Splash',
  'Liana',
  'Celerio',
  'Alto',
  'Kizashi',
  'Baleno',
  'Across'],
  'Škoda': ['Octavia',
  'Fabia',
  'Superb',
  'Rapid',
  'Scala',
  'Kodiaq',
  'Yeti',
  'Roomster',
  'Karoq',
  'Citigo',
  'Kamiq',
  'Felicia',
  'Favorit'],
 'Tesla': ['Model S', 'Model 3'],
 'Toyota': ['Corolla',
  'Yaris',
  'Rav4',
  'Avensis',
  'Auris',
  'C-HR',
  'Proace',
  'Aygo',
  'Land Cruiser',
  'Verso',
  'Corolla Verso',
  'Hilux',
  'Camry',
  'Proace City Verso',
  'Prius',
  'Proace City',
  'Urban Cruiser',
  'Supra',
  'Sienna',
  'Celica',
  'Avensis Verso',
  'Previa',
  'iQ',
  'GT86',
  '4Runner',
  'MR2',
  'Highlander'],
 'Volkswagen': ['Passat',
  'Golf',
  'Touran',
  'Tiguan',
  'Caddy',
  'Sharan',
  'Polo',
  'Golf Plus',
  'Touareg',
  'Transporter',
  'Multivan',
  'Caravelle',
  'Golf Sportsvan',
  'Arteon',
  'CC',
  'Passat CC',
  'Up!',
  'T-Roc',
  'New Beetle',
  'Amarok',
  'Scirocco',
  'Jetta',
  'Fox',
  'Eos',
  'Bora',
  'Phaeton',
  'T-Cross',
  'Lupo',
  'Tiguan Allspace',
  'California',
  'Beetle',
  'ID.3'],
 'Volvo': ['XC60',
  'XC90',
  'V60',
  'V90',
  'XC40',
  'V40',
  'V70',
  'S60',
  'V50',
  'XC70',
  'S80',
  'S90',
  'C30',
  'S40',
  'C70']};






